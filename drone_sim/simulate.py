#!/usr/bin/env python3
"""Simulación MuJoCo de un cuadricóptero para probar filtros de Kalman.

Publica IMU de 9 ejes, corriente de los cuatro BLDC y altímetro, con verdad
al lado para puntuar el filtro. Los comandos entran por un socket REP.

  python simulate.py
  python simulate.py --headless --accel-noise 0.2 --gyro-offset 0.01,0,0
  python simulate.py --config sensor_profile.example.json --log vuelo.jsonl
  python send_cmd.py --mode position --x 1 --y 0 --z 1.5 --yaw 0.4
  python view_sensors.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import zmq

import flight_control
import sensor_model
from flight_control import MOTOR_LABELS, START_POS, FlightControl
from sensor_model import FORMULA, AxisModel

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "quadcopter.xml"

HELP = """
Modelo de cada sensor (por eje, en las unidades de la medida):
  y = offset + scale * x + nonlin * x * |x| + N(0, noise)

noise es la desviación estándar de cada muestra, a --sensor-hz.
En un Kalman discreto, R = diag(noise**2). Un solo número se repite en todos
los ejes: --accel-noise 0.1 equivale a 0.1,0.1,0.1.

Marco del cuerpo (FLU): x adelante, y a la izquierda, z arriba.
Mundo: x, y, z arriba. Motores: FR, FL, RL, RR.
El acelerómetro es fuerza específica (en reposo, nivelado, lee +9.81 en z).
El magnetómetro está en µT. El altímetro es z del mundo, en metros.
La corriente es I = 0.40 A + 1.10 A/N * empuje, con retraso de primer orden.

Mensaje PUB (JSON), type=sample. El filtro debe leer "meas". "truth" es para puntuar.
  meas.accel[3], meas.gyro[3], meas.mag[3]     rad/s, m/s^2, µT, marco del cuerpo
  meas.current[4]                              amperios, orden FR FL RL RR
  meas.altimeter                               metros
  truth.pos, truth.vel, truth.quat_wxyz       estado
  truth.accel, truth.gyro, truth.mag, truth.current, truth.altimeter

Comandos, socket REQ (el valor se queda activo hasta el siguiente):
  {"mode":"auto"}
  {"mode":"position","x":1,"y":0,"z":1.5,"yaw":0.4}
  {"mode":"velocity","vx":0.3,"vy":0,"vz":0,"yaw_rate":0.2}
  {"mode":"attitude","roll":0.1,"pitch":0,"yaw_rate":0,"thrust":12.8}

pitch > 0 = morro arriba. roll > 0 = alabeo a la derecha (ala derecha abajo).
thrust es el empuje total en newtons. Si se omite, se usa el de hover.
En macOS la ventana se reabre sola con mjpython. --headless la omite.
"""

SENSOR_DEFAULTS = {
    "seed": 1,
    "sensor_hz": 200.0,
    "accel_noise": [0.05, 0.05, 0.08],
    "accel_offset": [0.0, 0.0, 0.0],
    "accel_scale": [1.0, 1.0, 1.0],
    "accel_nonlin": [0.0, 0.0, 0.0],
    "gyro_noise": [0.004, 0.004, 0.005],
    "gyro_offset": [0.0, 0.0, 0.0],
    "gyro_scale": [1.0, 1.0, 1.0],
    "gyro_nonlin": [0.0, 0.0, 0.0],
    "mag_noise": [0.4, 0.4, 0.6],
    "mag_offset": [0.0, 0.0, 0.0],
    "mag_scale": [1.0, 1.0, 1.0],
    "mag_nonlin": [0.0, 0.0, 0.0],
    "current_noise": [0.05, 0.05, 0.05, 0.05],
    "current_offset": [0.0, 0.0, 0.0, 0.0],
    "current_scale": [1.0, 1.0, 1.0, 1.0],
    "current_nonlin": [0.0, 0.0, 0.0, 0.0],
    "alt_noise": 0.05,
    "alt_offset": 0.0,
    "alt_scale": 1.0,
    "alt_nonlin": 0.0,
}


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def csv_floats(n: int):
    def parse(text: str):
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
        try:
            vals = [float(p) for p in parts]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"no es una lista de números: {text}") from exc
        if len(vals) == 1 and n > 1:
            vals = vals * n
        if len(vals) != n:
            raise argparse.ArgumentTypeError(f"hacen falta {n} números separados por coma, o uno solo")
        return vals

    return parse


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Banco MuJoCo de un cuadricóptero para filtros de Kalman.",
        formatter_class=HelpFormatter,
        epilog=HELP,
    )
    p.add_argument("--config", help="JSON con parámetros de sensor. La línea de comandos pisa el archivo.")
    p.add_argument("--headless", action="store_true", help="sin la ventana de MuJoCo")
    p.add_argument("--fast", action="store_true", help="correr lo más rápido posible, sin esperar al reloj")
    p.add_argument("--seconds", type=float, default=0.0, help="detener a este tiempo de simulación. 0 = hasta Ctrl+C")
    p.add_argument("--log", type=Path, help="guardar cada muestra en JSONL, además de ZMQ")
    p.add_argument("--pub-bind", default="tcp://127.0.0.1:5555", help="PUB de medidas")
    p.add_argument("--cmd-bind", default="tcp://127.0.0.1:5556", help="REP de comandos")
    p.add_argument("--mode", choices=("auto", "position", "velocity", "attitude"), default="auto")
    p.add_argument("--seed", type=int, default=defaults["seed"])
    p.add_argument("--sensor-hz", type=float, default=defaults["sensor_hz"], dest="sensor_hz")

    def add_vec(flag: str, dest: str, n: int):
        meta = "X,Y,Z" if n == 3 else "FR,FL,RL,RR"
        p.add_argument(flag, dest=dest, type=csv_floats(n), default=defaults[dest], metavar=meta)

    add_vec("--accel-noise", "accel_noise", 3)
    add_vec("--accel-offset", "accel_offset", 3)
    add_vec("--accel-scale", "accel_scale", 3)
    add_vec("--accel-nonlin", "accel_nonlin", 3)
    add_vec("--gyro-noise", "gyro_noise", 3)
    add_vec("--gyro-offset", "gyro_offset", 3)
    add_vec("--gyro-scale", "gyro_scale", 3)
    add_vec("--gyro-nonlin", "gyro_nonlin", 3)
    add_vec("--mag-noise", "mag_noise", 3)
    add_vec("--mag-offset", "mag_offset", 3)
    add_vec("--mag-scale", "mag_scale", 3)
    add_vec("--mag-nonlin", "mag_nonlin", 3)
    add_vec("--current-noise", "current_noise", 4)
    add_vec("--current-offset", "current_offset", 4)
    add_vec("--current-scale", "current_scale", 4)
    add_vec("--current-nonlin", "current_nonlin", 4)
    p.add_argument("--alt-noise", dest="alt_noise", type=float, default=defaults["alt_noise"])
    p.add_argument("--alt-offset", dest="alt_offset", type=float, default=defaults["alt_offset"])
    p.add_argument("--alt-scale", dest="alt_scale", type=float, default=defaults["alt_scale"])
    p.add_argument("--alt-nonlin", dest="alt_nonlin", type=float, default=defaults["alt_nonlin"])
    return p


def relaunch_with_mjpython(argv) -> bool:
    """En macOS el visor solo arranca con el lanzador mjpython.

    Si hace falta, sustituye este proceso. Devuelve False cuando no hay visor
    posible y la simulación debe seguir sin ventana.
    """
    if sys.platform != "darwin" or os.environ.get("MJPYTHON_BIN"):
        return True
    # No usar resolve(): en un venv sys.executable es un enlace al Python real,
    # y mjpython está al lado del enlace, no del binario.
    launcher = Path(sys.executable).parent / "mjpython"
    if not launcher.is_file():
        print(
            "En macOS el visor de MuJoCo necesita mjpython y no está en este entorno. Sigo sin ventana.",
            file=sys.stderr,
        )
        return False
    forwarded = list(sys.argv) if argv is None else [sys.argv[0], *argv]
    os.execv(str(launcher), [str(launcher), *forwarded])
    return False


def parse_args(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    defaults = dict(SENSOR_DEFAULTS)
    if known.config:
        path = Path(known.config)
        with path.open(encoding="utf-8") as fh:
            loaded = json.load(fh)
        if not isinstance(loaded, dict):
            raise SystemExit(f"{path} tiene que ser un objeto JSON")
        loaded = {k: v for k, v in loaded.items() if not str(k).startswith("_")}
        unknown = sorted(set(loaded) - set(SENSOR_DEFAULTS))
        if unknown:
            raise SystemExit(f"claves desconocidas en {path}: {', '.join(unknown)}")
        defaults.update(loaded)
    return build_parser(defaults).parse_args(argv)


def make_sensors(args: argparse.Namespace) -> dict[str, AxisModel]:
    specs = (
        ("accel", 3, "accel"),
        ("gyro", 3, "gyro"),
        ("mag", 3, "mag"),
        ("current", 4, "current"),
        ("altimeter", 1, "alt"),
    )
    models = {}
    for name, n, prefix in specs:
        models[name] = AxisModel(
            noise=getattr(args, f"{prefix}_noise"),
            offset=getattr(args, f"{prefix}_offset"),
            scale=getattr(args, f"{prefix}_scale"),
            nonlin=getattr(args, f"{prefix}_nonlin"),
            n=n,
            name=name,
        )
    return models


def sensor_slice(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise RuntimeError(f"el modelo no tiene el sensor '{name}'")
    adr = int(model.sensor_adr[sid])
    dim = int(model.sensor_dim[sid])
    return np.array(data.sensordata[adr : adr + dim], dtype=float)


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (str, int)) or obj is None:
        return obj
    raise TypeError(type(obj))


def config_message(args, controller: FlightControl, sensors, mag_ref, dt) -> dict:
    return {
        "type": "config",
        "formula": FORMULA,
        "noise_meaning": "desviacion estandar de cada muestra. R_kalman = diag(noise**2)",
        "frame_body": "FLU: x adelante, y izquierda, z arriba",
        "frame_world": "x, y, z arriba",
        "motor_order": list(MOTOR_LABELS),
        "sensor_dt": dt,
        "seed": args.seed,
        "mass_kg": controller.mass,
        "gravity_m_s2": controller.gravity.tolist(),
        "mag_ref_world_uT": mag_ref.tolist(),
        "motor_model": {
            "I_A": "I_idle + k_i * thrust_N",
            "I_idle_A": controller.i_idle,
            "k_i_A_per_N": controller.k_i,
            "tau_s": controller.tau,
            "note": "el empuje de cada motor sigue al comando con retardo de primer orden",
        },
        "sensors": {name: model.to_dict() for name, model in sensors.items()},
        "units": {
            "accel": "m/s^2",
            "gyro": "rad/s",
            "mag": "uT",
            "current": "A",
            "altimeter": "m",
        },
    }


def print_banner(args, controller, sensors, physics_hz, sensor_hz) -> None:
    clock = "lo más rápido posible" if args.fast else "tiempo real"
    print("Banco de pruebas — cuadricóptero MuJoCo")
    print(
        f"  masa {controller.mass:.3f} kg   física {physics_hz:.0f} Hz   "
        f"sensores {sensor_hz:.1f} Hz   semilla {args.seed}"
    )
    print(f"  medidas   PUB  {args.pub_bind}")
    print(f"  comandos  REP  {args.cmd_bind}")
    print(f"  reloj: {clock}    modo inicial: {args.mode}")
    if args.log:
        print(f"  registro: {args.log}")
    print(f"  {FORMULA}")
    print("  noise = desviación estándar de cada muestra; R = diag(noise²) en un Kalman discreto")
    for model in sensors.values():
        print(f"  {model.summary()}")
    print("  rutina: hover, cuadrado, trepada y descenso, en bucle. Ctrl+C para salir.")
    print()


def open_sockets(args):
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.setsockopt(zmq.SNDHWM, 1000)
    cmd = ctx.socket(zmq.REP)
    cmd.setsockopt(zmq.LINGER, 0)
    try:
        pub.bind(args.pub_bind)
        cmd.bind(args.cmd_bind)
    except zmq.ZMQError as exc:
        pub.close(0)
        cmd.close(0)
        ctx.term()
        raise SystemExit(f"no se pudo abrir ZMQ ({exc}). ¿Ya hay una simulación en ese puerto?") from exc
    return ctx, pub, cmd


def poll_command(sock, controller: FlightControl) -> None:
    try:
        raw = sock.recv(flags=zmq.NOBLOCK)
    except zmq.Again:
        return
    try:
        msg = json.loads(raw.decode())
        info = controller.submit_command(msg)
        reply = {"ok": True, **clean(info)}
        print(f"Comando -> {json.dumps(reply, ensure_ascii=False)}")
    except Exception as exc:
        reply = {"ok": False, "error": str(exc)}
        print(f"Comando rechazado: {exc}")
    sock.send_json(reply)


def publish(sock, payload: dict) -> None:
    try:
        sock.send_json(clean(payload), flags=zmq.NOBLOCK)
    except zmq.Again:
        pass


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.headless and not relaunch_with_mjpython(argv):
        args.headless = True
    if args.sensor_hz <= 0 or args.sensor_hz > 1000:
        raise SystemExit("--sensor-hz tiene que estar entre 0 y 1000")
    if args.seconds < 0:
        raise SystemExit("--seconds no puede ser negativo")

    sensors = make_sensors(args)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    if model.nq != 7 or model.nv != 6:
        raise SystemExit(f"se esperaba un solo freejoint (nq=7, nv=6) y el modelo tiene nq={model.nq} nv={model.nv}")

    controller = FlightControl(model)
    physics_dt = float(model.opt.timestep)
    substeps = max(1, int(round((1.0 / args.sensor_hz) / physics_dt)))
    sensor_dt = substeps * physics_dt
    sensor_hz = 1.0 / sensor_dt
    if abs(sensor_hz - args.sensor_hz) / args.sensor_hz > 0.02:
        print(
            f"aviso: --sensor-hz {args.sensor_hz:g} no cabe exacto en el paso de física; se usa {sensor_hz:.2f} Hz",
            file=sys.stderr,
        )

    data.qpos[:] = 0.0
    data.qpos[0:3] = START_POS
    data.qpos[3] = 1.0
    data.qvel[:] = 0.0
    data.ctrl[controller.ctrl_ids] = controller.thrust
    mujoco.mj_forward(model, data)

    if args.mode != "auto":
        controller.submit_command(
            {"mode": args.mode, "x": 0.0, "y": 0.0, "z": float(START_POS[2]), "yaw": 0.0, "thrust": controller.hover_thrust}
        )

    mag_ref = np.array(model.opt.magnetic, dtype=float)
    rng = np.random.default_rng(args.seed)
    print_banner(args, controller, sensors, 1.0 / physics_dt, sensor_hz)

    ctx, pub, cmd = open_sockets(args)
    log_fh = args.log.open("w", encoding="utf-8") if args.log else None
    cfg = config_message(args, controller, sensors, mag_ref, sensor_dt)
    publish(pub, cfg)
    if log_fh:
        log_fh.write(json.dumps(clean(cfg), ensure_ascii=False) + "\n")

    viewer = None
    seq = 0
    sat_steps = 0
    window_steps = 0
    next_status = 2.0
    wall_next = time.perf_counter()
    try:
        if not args.headless:
            import mujoco.viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(model, data)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = controller.body_id
            viewer.cam.distance = 3.4
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.lookat[:] = START_POS

        while viewer is None or viewer.is_running():
            if args.seconds and float(data.time) >= args.seconds:
                break
            poll_command(cmd, controller)
            output = None
            for _ in range(substeps):
                output = controller.step(data, physics_dt)
                mujoco.mj_step(model, data)
                window_steps += 1
                if output["mixer_scale"] < 0.999:
                    sat_steps += 1

            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                print("El estado numérico dejó de ser finito. Paro.")
                return 1

            assert output is not None
            accel = sensor_slice(model, data, "accel")
            gyro = sensor_slice(model, data, "gyro")
            mag = sensor_slice(model, data, "mag")
            pos = sensor_slice(model, data, "pos")
            current = np.array(output["current"], dtype=float)

            seq += 1
            sample = {
                "type": "sample",
                "seq": seq,
                "t": float(data.time),
                "dt": sensor_dt,
                "mode": output["mode"],
                "meas": {
                    "accel": sensors["accel"].apply(accel, rng),
                    "gyro": sensors["gyro"].apply(gyro, rng),
                    "mag": sensors["mag"].apply(mag, rng),
                    "current": sensors["current"].apply(current, rng),
                    "altimeter": sensors["altimeter"].apply(pos[2], rng),
                },
                "truth": {
                    "pos": pos,
                    "vel": np.array(data.qvel[0:3], dtype=float),
                    "quat_wxyz": np.array(data.qpos[3:7], dtype=float),
                    "accel": accel,
                    "gyro": gyro,
                    "mag": mag,
                    "current": current,
                    "altimeter": float(pos[2]),
                    "thrust": output["thrust"],
                },
                "setpoint": {"pos": output["pos_sp"], "yaw": output["yaw_sp"]},
            }
            publish(pub, sample)
            if log_fh:
                log_fh.write(json.dumps(clean(sample), ensure_ascii=False) + "\n")

            if float(data.time) >= next_status:
                err = float(np.linalg.norm(pos - output["pos_sp"]))
                sat = 100.0 * sat_steps / max(window_steps, 1)
                yaw = flight_control.yaw_of(flight_control.mat_from_quat(data.qpos[3:7]))
                print(
                    f"t={data.time:6.1f}s  modo={output['mode']:9}  "
                    f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f})  "
                    f"error={err:5.2f} m  yaw={yaw:6.2f}  saturación={sat:4.0f}%"
                )
                next_status += 2.0
                sat_steps = 0
                window_steps = 0
                publish(pub, cfg)

            if viewer is not None and seq % max(1, int(round(sensor_hz / 60.0))) == 0:
                viewer.sync()

            if not args.fast:
                wall_next += sensor_dt
                delay = wall_next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    wall_next = time.perf_counter()
    except KeyboardInterrupt:
        print("\nDetenido.")
    finally:
        if log_fh:
            log_fh.close()
        if viewer is not None:
            viewer.close()
        pub.close(0)
        cmd.close(0)
        ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

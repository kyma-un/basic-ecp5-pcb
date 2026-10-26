"""Control del cuadricóptero: rutina, lazo de posición/actitud y mezcla de motores.

El lazo usa el estado real de MuJoCo. Las medidas ruidosas salen por ZMQ para
el filtro de Kalman; el piloto automático de este banco no las consume.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Orden de publicación y de data.ctrl. Tiene que coincidir con el XML.
ACTUATORS = ("fr", "fl", "rl", "rr")
MOTOR_LABELS = ("FR", "FL", "RL", "RR")

START_POS = np.array([0.0, 0.0, 1.0])

# Ganancias sobre el doble integrador (m/s² por m, y por m/s).
KP_POS = np.array([2.0, 2.0, 5.5])
KD_POS = np.array([2.4, 2.4, 4.0])
# Error de SO(3) en el cuerpo, N·m por radián, y amortiguación sobre ω.
KP_ATT = np.array([0.55, 0.55, 0.10])
KD_ATT = np.array([0.16, 0.16, 0.07])

MAX_TILT_RAD = np.deg2rad(28.0)
V_MAX = 0.8
VZ_MAX = 0.65
YAW_RATE_MAX = 0.5

ROLL_LIMIT = 0.6
PITCH_LIMIT = 0.6
YAW_RATE_CMD_LIMIT = 2.0
V_CMD_LIMIT = 3.0
Z_MIN = 0.25
Z_MAX = 8.0
XY_LIMIT = 20.0

DRAG_N_PER_MPS = 0.28
MOTOR_TAU_S = 0.03
I_IDLE_A = 0.40
K_I_A_PER_N = 1.10


def wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def mat_from_quat(q) -> np.ndarray:
    out = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(out, np.ascontiguousarray(q, dtype=np.float64))
    return out.reshape(3, 3)


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Cuerpo→mundo. pitch > 0 = morro arriba. roll > 0 = ala derecha abajo."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    # Ry(-pitch): morro arriba lleva +z del cuerpo hacia -x del mundo.
    ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def so3_error(R: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Error de actitud en el marco del cuerpo (vee del mapa de SO(3))."""
    skew = R_des.T @ R - R.T @ R_des
    return 0.5 * np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


class AutoRoutine:
    """Waypoints en bucle. El slew del piloto suaviza los saltos."""

    def __init__(self):
        # t_fin, posición mundo, yaw (rad, sentido antihorario visto desde arriba)
        raw = (
            (4.0, (0.0, 0.0, 1.0), 0.0),       # hover: sesgo observable
            (9.0, (0.0, 0.0, 1.7), 0.0),
            (13.0, (0.0, 0.0, 1.7), 0.0),
            (19.0, (2.0, 0.0, 1.7), 0.0),
            (25.0, (2.0, 2.0, 1.7), 1.1),
            (31.0, (0.0, 2.0, 1.7), 2.2),
            (37.0, (0.0, 0.0, 1.7), 0.3),
            (42.0, (0.0, 0.0, 2.5), 0.0),      # trepada
            (48.0, (0.0, -1.5, 0.7), -0.6),    # descenso en movimiento
            (54.0, (0.0, 0.0, 1.0), 0.0),
        )
        self.segments = tuple((t, np.array(p, dtype=float), float(y)) for t, p, y in raw)
        prev = 0.0
        for t_end, _, _ in self.segments:
            if t_end <= prev:
                raise RuntimeError("la rutina tiene tiempos desordenados")
            prev = t_end
        self.duration = self.segments[-1][0]

    def goal(self, t: float) -> tuple[np.ndarray, float]:
        tau = 0.0 if t <= 0.0 else t % self.duration
        for t_end, pos, yaw in self.segments:
            if tau <= t_end:
                return pos.copy(), yaw
        pos, yaw = self.segments[-1][1], self.segments[-1][2]
        return pos.copy(), yaw


class FlightControl:
    def __init__(self, model: mujoco.MjModel):
        _check_math()
        self.model = model
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self.body_id < 0:
            raise RuntimeError("el XML no tiene el cuerpo 'base'")
        self.mass = float(model.body_subtreemass[self.body_id])
        self.gravity = np.array(model.opt.gravity, dtype=float)
        gnorm = float(np.linalg.norm(self.gravity))
        if gnorm < 1e-6:
            raise RuntimeError("la gravedad del modelo es nula")
        self.up = -self.gravity / gnorm
        self.g = gnorm
        self.hover_thrust = self.mass * self.g

        self.ctrl_ids = []
        columns = []
        for name in ACTUATORS:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if aid < 0 or sid < 0:
                raise RuntimeError(f"faltan el actuador o el sitio '{name}'")
            gear = np.array(model.actuator_gear[aid], dtype=float)
            if abs(gear[2] - 1.0) > 1e-9 or np.max(np.abs(gear[[0, 1, 3, 4]])) > 1e-9:
                raise RuntimeError(f"el actuador '{name}' debe tener gear = 0 0 1 0 0 k")
            x, y, _z = model.site_pos[sid]
            # F = (0,0,f) en el cuerpo → par = r × F = (y f, -x f, k f)
            columns.append([1.0, y, -x, gear[5]])
            self.ctrl_ids.append(aid)
        self.A = np.column_stack(columns)
        cond = float(np.linalg.cond(self.A))
        if cond > 1e8 or not np.isfinite(cond):
            raise RuntimeError(f"la mezcla de motores está mal condicionada (cond={cond})")
        self.fmax = float(min(model.actuator_ctrlrange[i, 1] for i in self.ctrl_ids))
        self._check_mixer()

        self.routine = AutoRoutine()
        self.mode = "auto"
        self.pos_goal = START_POS.copy()
        self.yaw_goal = 0.0
        self.pos_sp = START_POS.copy()
        self.yaw_sp = 0.0
        self.v_ff = np.zeros(3)
        self.v_cmd = np.zeros(3)
        self.yaw_rate_cmd = 0.0
        self.roll_cmd = 0.0
        self.pitch_cmd = 0.0
        self.thrust_cmd = self.hover_thrust
        self.last_pos = START_POS.copy()
        self.last_yaw = 0.0
        self.thrust = np.full(4, self.hover_thrust / 4.0)
        self.mixer_scale = 1.0
        self.tau = MOTOR_TAU_S
        self.i_idle = I_IDLE_A
        self.k_i = K_I_A_PER_N

    def _check_mixer(self) -> None:
        hover = np.array([self.hover_thrust, 0.0, 0.0, 0.0])
        f = np.linalg.solve(self.A, hover)
        if not np.allclose(f, self.hover_thrust / 4.0, atol=1e-8):
            raise RuntimeError(f"el hover no reparte igual entre motores: {f}")
        wrench = np.array([self.hover_thrust, 0.05, -0.04, 0.01])
        back = self.A @ np.linalg.solve(self.A, wrench)
        if not np.allclose(back, wrench, atol=1e-8):
            raise RuntimeError("la mezcla no recupera el wrench pedido")

    def submit_command(self, cmd: dict) -> dict:
        if not isinstance(cmd, dict):
            raise ValueError("el comando debe ser un objeto JSON")
        mode = str(cmd.get("mode", "")).lower()
        notes: list[str] = []
        entering = mode != self.mode

        if mode == "auto":
            if entering:
                self.pos_sp = self.last_pos.copy()
                self.yaw_sp = self.last_yaw
            self.mode = "auto"
        elif mode == "position":
            x = _finite(cmd, "x")
            y = _finite(cmd, "y")
            z = _finite(cmd, "z")
            if abs(x) > XY_LIMIT or abs(y) > XY_LIMIT:
                raise ValueError(f"x e y deben estar entre ±{XY_LIMIT:.0f} m")
            z_clamped = float(np.clip(z, Z_MIN, Z_MAX))
            if z_clamped != z:
                notes.append(f"z limitado a [{Z_MIN}, {Z_MAX}] m")
            if entering:
                self.pos_sp = self.last_pos.copy()
                self.yaw_sp = self.last_yaw
            self.pos_goal = np.array([x, y, z_clamped])
            if "yaw" in cmd and cmd["yaw"] is not None:
                self.yaw_goal = _finite(cmd, "yaw")
            elif entering:
                self.yaw_goal = self.last_yaw
            self.mode = "position"
        elif mode == "velocity":
            vx = _clip_opt(cmd, "vx", V_CMD_LIMIT, notes, "vx")
            vy = _clip_opt(cmd, "vy", V_CMD_LIMIT, notes, "vy")
            vz = _clip_opt(cmd, "vz", V_CMD_LIMIT, notes, "vz")
            yaw_rate = _clip_opt(cmd, "yaw_rate", YAW_RATE_CMD_LIMIT, notes, "yaw_rate")
            if entering:
                self.yaw_sp = self.last_yaw
            self.v_cmd = np.array([vx, vy, vz])
            self.yaw_rate_cmd = yaw_rate
            self.mode = "velocity"
        elif mode == "attitude":
            roll = _clip_opt(cmd, "roll", ROLL_LIMIT, notes, "roll")
            pitch = _clip_opt(cmd, "pitch", PITCH_LIMIT, notes, "pitch")
            yaw_rate = _clip_opt(cmd, "yaw_rate", YAW_RATE_CMD_LIMIT, notes, "yaw_rate")
            if "thrust" in cmd and cmd["thrust"] is not None:
                thrust = _finite(cmd, "thrust")
                thrust = float(np.clip(thrust, 0.0, 4.0 * self.fmax))
                if thrust != float(cmd["thrust"]):
                    notes.append("thrust limitado al rango de los motores")
                self.thrust_cmd = thrust
            elif entering:
                self.thrust_cmd = self.hover_thrust
            if entering:
                self.yaw_sp = self.last_yaw
            self.roll_cmd = roll
            self.pitch_cmd = pitch
            self.yaw_rate_cmd = yaw_rate
            self.mode = "attitude"
        else:
            raise ValueError("mode debe ser auto, position, velocity o attitude")

        info = self.command_state()
        if notes:
            info["note"] = "; ".join(notes)
        return info

    def command_state(self) -> dict:
        state = {"mode": self.mode}
        if self.mode in ("auto", "position"):
            state["pos_goal"] = self.pos_goal.tolist()
            state["yaw_goal"] = self.yaw_goal
        elif self.mode == "velocity":
            state["v_cmd"] = self.v_cmd.tolist()
            state["yaw_rate"] = self.yaw_rate_cmd
        elif self.mode == "attitude":
            state["roll"] = self.roll_cmd
            state["pitch"] = self.pitch_cmd
            state["yaw_rate"] = self.yaw_rate_cmd
            state["thrust"] = self.thrust_cmd
        return state

    def step(self, data: mujoco.MjData, dt: float) -> dict:
        pos = np.array(data.qpos[0:3], dtype=float)
        R = mat_from_quat(data.qpos[3:7])
        vel = np.array(data.qvel[0:3], dtype=float)
        omega = np.array(data.qvel[3:6], dtype=float)
        yaw = yaw_of(R)
        self.last_pos = pos
        self.last_yaw = yaw

        if self.mode == "auto":
            self.pos_goal, self.yaw_goal = self.routine.goal(float(data.time))
            total, R_des = self._position_law(pos, vel, R, dt)
        elif self.mode == "position":
            total, R_des = self._position_law(pos, vel, R, dt)
        elif self.mode == "velocity":
            self.yaw_sp = wrap(self.yaw_sp + self.yaw_rate_cmd * dt)
            a_des = KD_POS * (self.v_cmd - vel) - self.gravity
            a_des = self._limit_accel(a_des)
            total = self._thrust_along_body(a_des, R)
            R_des = self._heading_attitude(a_des, self.yaw_sp)
            self.v_ff = self.v_cmd.copy()
        elif self.mode == "attitude":
            self.yaw_sp = wrap(self.yaw_sp + self.yaw_rate_cmd * dt)
            total = self.thrust_cmd
            R_des = rotation_matrix(self.roll_cmd, self.pitch_cmd, self.yaw_sp)
        else:
            raise RuntimeError(self.mode)

        torque = -KP_ATT * so3_error(R, R_des) - KD_ATT * omega
        cmd, scale = self._mix(total, torque)
        self.mixer_scale = scale
        alpha = 1.0 - np.exp(-dt / self.tau)
        self.thrust = self.thrust + alpha * (cmd - self.thrust)
        self.thrust = np.clip(self.thrust, 0.0, self.fmax)
        current = self.i_idle + self.k_i * self.thrust

        data.xfrc_applied[self.body_id, :] = 0.0
        data.xfrc_applied[self.body_id, 0:3] = -DRAG_N_PER_MPS * data.qvel[0:3]
        data.ctrl[self.ctrl_ids] = self.thrust

        return {
            "thrust_cmd": cmd,
            "thrust": self.thrust.copy(),
            "current": current.copy(),
            "mode": self.mode,
            "pos_sp": self.pos_sp.copy(),
            "yaw_sp": self.yaw_sp,
            "mixer_scale": scale,
        }

    def _position_law(self, pos, vel, R, dt):
        prev = self.pos_sp.copy()
        self.pos_sp = self._slew_pos(self.pos_sp, self.pos_goal, dt)
        self.v_ff = (self.pos_sp - prev) / dt
        yaw_err = wrap(self.yaw_goal - self.yaw_sp)
        yaw_step = float(np.clip(yaw_err, -YAW_RATE_MAX * dt, YAW_RATE_MAX * dt))
        self.yaw_sp = wrap(self.yaw_sp + yaw_step)
        pos_err = np.clip(self.pos_sp - pos, -2.0, 2.0)
        a_des = KP_POS * pos_err + KD_POS * (self.v_ff - vel) - self.gravity
        a_des = self._limit_accel(a_des)
        total = self._thrust_along_body(a_des, R)
        return total, self._heading_attitude(a_des, self.yaw_sp)

    def _slew_pos(self, pos_sp: np.ndarray, goal: np.ndarray, dt: float) -> np.ndarray:
        delta = goal - pos_sp
        horiz = float(np.linalg.norm(delta[:2]))
        max_h = V_MAX * dt
        if horiz > max_h > 0.0:
            delta = delta.copy()
            delta[:2] *= max_h / horiz
        delta = delta.copy()
        delta[2] = float(np.clip(delta[2], -VZ_MAX * dt, VZ_MAX * dt))
        return pos_sp + delta

    def _limit_accel(self, a_des: np.ndarray) -> np.ndarray:
        along = float(np.dot(a_des, self.up))
        lateral = a_des - along * self.up
        lat = float(np.linalg.norm(lateral))
        max_lat = self.g * np.tan(MAX_TILT_RAD)
        if lat > max_lat > 0.0:
            lateral *= max_lat / lat
        along = float(np.clip(along, 0.35 * self.g, 1.7 * self.g))
        return lateral + along * self.up

    def _thrust_along_body(self, a_des: np.ndarray, R: np.ndarray) -> float:
        # Empuje que produce a_des proyectado sobre el eje actual del cuerpo.
        return float(np.clip(self.mass * np.dot(a_des, R[:, 2]), 0.0, 4.0 * self.fmax))

    def _heading_attitude(self, a_des: np.ndarray, yaw: float) -> np.ndarray:
        z_d = a_des / max(float(np.linalg.norm(a_des)), 1e-6)
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_d = np.cross(z_d, x_c)
        n = float(np.linalg.norm(y_d))
        if n < 1e-6:
            x_c = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
            y_d = np.cross(z_d, x_c)
            n = float(np.linalg.norm(y_d))
        y_d = y_d / n
        x_d = np.cross(y_d, z_d)
        return np.column_stack((x_d, y_d, z_d))

    def _mix(self, total_thrust: float, torque: np.ndarray) -> tuple[np.ndarray, float]:
        total_thrust = float(np.clip(total_thrust, 0.0, 4.0 * self.fmax))
        base = np.array([total_thrust, 0.0, 0.0, 0.0])
        delta = np.array([0.0, torque[0], torque[1], torque[2]])
        lo, hi = 0.0, 1.0
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            forces = np.linalg.solve(self.A, base + mid * delta)
            if np.all(forces >= -1e-8) and np.all(forces <= self.fmax + 1e-8):
                lo = mid
            else:
                hi = mid
        forces = np.linalg.solve(self.A, base + lo * delta)
        return np.clip(forces, 0.0, self.fmax), lo


def _finite(cmd: dict, key: str) -> float:
    if key not in cmd or cmd[key] is None:
        raise ValueError(f"falta '{key}'")
    value = float(cmd[key])
    if not np.isfinite(value):
        raise ValueError(f"'{key}' no es un número finito")
    return value


def _clip_opt(cmd: dict, key: str, limit: float, notes: list[str], label: str) -> float:
    if key not in cmd or cmd[key] is None:
        return 0.0
    value = _finite(cmd, key)
    clipped = float(np.clip(value, -limit, limit))
    if clipped != value:
        notes.append(f"{label} limitado a ±{limit:g}")
    return clipped


def _check_math() -> None:
    R = rotation_matrix(0.0, 0.2, 0.0)
    if R[2, 0] < 0.1 or R[0, 2] > -0.1:
        raise RuntimeError("pitch positivo no levanta el morro")
    R = rotation_matrix(0.2, 0.0, 0.0)
    if R[2, 1] < 0.1:
        raise RuntimeError("roll positivo no baja el lado derecho")
    R = rotation_matrix(0.0, 0.0, np.pi / 2.0)
    if R[1, 0] < 0.9 or abs(R[0, 0]) > 0.1:
        raise RuntimeError("yaw positivo no gira hacia +y")
    if abs(so3_error(np.eye(3), np.eye(3))).max() > 1e-12:
        raise RuntimeError("el error de actitud no es cero en la identidad")
    c, s = np.cos(0.1), np.sin(0.1)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    err = so3_error(rx, np.eye(3))
    if abs(err[0] - s) > 1e-9 or abs(err[1]) > 1e-9 or abs(err[2]) > 1e-9:
        raise RuntimeError(f"signo del error de actitud inesperado: {err}")
    if abs(wrap(np.pi + 0.1) - (-np.pi + 0.1)) > 1e-9:
        raise RuntimeError("wrap de ángulos falló")

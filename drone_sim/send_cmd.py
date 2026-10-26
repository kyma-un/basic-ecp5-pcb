#!/usr/bin/env python3
"""Manda un comando de vuelo a la simulación y espera el acuse.

  python send_cmd.py --mode auto
  python send_cmd.py --mode position --x 1 --y -0.5 --z 1.8 --yaw 0.4
  python send_cmd.py --mode velocity --vx 0.4 --yaw-rate 0.3
  python send_cmd.py --mode attitude --roll 0.1 --pitch 0 --thrust 13
  python send_cmd.py --json '{"mode":"auto"}'
"""

from __future__ import annotations

import argparse
import json
import sys

import zmq


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Envía un comando ZMQ al cuadricóptero.")
    p.add_argument("--connect", default="tcp://127.0.0.1:5556", help="socket REP de simulate.py")
    p.add_argument("--timeout", type=float, default=2.0, help="segundos de espera del acuse")
    p.add_argument("--json", help="objeto JSON crudo. Si está, ignora el resto de campos.")
    p.add_argument("--mode", choices=("auto", "position", "velocity", "attitude"))
    p.add_argument("--x", type=float)
    p.add_argument("--y", type=float)
    p.add_argument("--z", type=float)
    p.add_argument("--yaw", type=float, help="radianes, antihorario visto desde arriba")
    p.add_argument("--vx", type=float)
    p.add_argument("--vy", type=float)
    p.add_argument("--vz", type=float)
    p.add_argument("--yaw-rate", dest="yaw_rate", type=float, help="rad/s")
    p.add_argument("--roll", type=float, help="rad. positivo = ala derecha abajo")
    p.add_argument("--pitch", type=float, help="rad. positivo = morro arriba")
    p.add_argument("--thrust", type=float, help="empuje total, N")
    return p


def command_from_args(args: argparse.Namespace) -> dict:
    if args.json:
        msg = json.loads(args.json)
        if not isinstance(msg, dict):
            raise SystemExit("--json tiene que ser un objeto")
        return msg
    if not args.mode:
        raise SystemExit("indica --mode o --json")
    msg: dict = {"mode": args.mode}
    for key in ("x", "y", "z", "yaw", "vx", "vy", "vz", "yaw_rate", "roll", "pitch", "thrust"):
        value = getattr(args, key)
        if value is not None:
            msg[key] = value
    return msg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    msg = command_from_args(args)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    timeout_ms = max(1, int(args.timeout * 1000))
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(args.connect)
    try:
        sock.send_json(msg)
        reply = sock.recv_json()
    except zmq.Again:
        print(
            f"Sin respuesta de {args.connect} en {args.timeout:.1f} s. ¿Está corriendo simulate.py?",
            file=sys.stderr,
        )
        return 1
    except zmq.ZMQError as exc:
        print(f"Error de ZMQ: {exc}", file=sys.stderr)
        return 1
    finally:
        sock.close(0)
        ctx.term()

    print(json.dumps(reply, ensure_ascii=False, indent=2))
    return 0 if reply.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

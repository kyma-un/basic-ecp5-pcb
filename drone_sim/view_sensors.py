#!/usr/bin/env python3
"""Muestra en vivo las medidas que publica simulate.py.

  python view_sensors.py
  python view_sensors.py --text
  python view_sensors.py --text --hz 10
  python view_sensors.py --no-truth --window 20

Continuo = medida. Punteado = verdad, si el mensaje la trae.
"""

from __future__ import annotations

import argparse
import json
from collections import deque

import zmq

COLORS = ("#d62728", "#2ca02c", "#1f77b4")
MOTOR_COLORS = ("#d62728", "#2ca02c", "#1f77b4", "#ff7f0e")
MOTOR_LABELS = ("FR", "FL", "RL", "RR")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualiza las medidas ZMQ del cuadricóptero.")
    p.add_argument("--connect", default="tcp://127.0.0.1:5555", help="PUB de simulate.py")
    p.add_argument("--window", type=float, default=12.0, help="segundos visibles en la gráfica")
    p.add_argument("--text", action="store_true", help="imprimir en la terminal en lugar de graficar")
    p.add_argument("--hz", type=float, default=5.0, help="con --text, impresiones por segundo. 0 = todas las muestras")
    p.add_argument("--no-truth", action="store_true", help="no dibujar la verdad encima de la medida")
    return p


def open_sub(endpoint: str):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVHWM, 4000)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.connect(endpoint)
    return ctx, sock


def format_sample(sample: dict) -> str:
    meas = sample["meas"]
    a, g, m = meas["accel"], meas["gyro"], meas["mag"]
    cur = meas["current"]
    return (
        f"t={sample['t']:7.2f}  alt={meas['altimeter']:7.3f} m  "
        f"a=[{a[0]:7.3f} {a[1]:7.3f} {a[2]:7.3f}]  "
        f"g=[{g[0]:7.3f} {g[1]:7.3f} {g[2]:7.3f}]  "
        f"m=[{m[0]:7.2f} {m[1]:7.2f} {m[2]:7.2f}]  "
        f"I=[{cur[0]:5.2f} {cur[1]:5.2f} {cur[2]:5.2f} {cur[3]:5.2f}]"
    )


def text_loop(sock, hz: float) -> None:
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    printed_config = False
    latest = None
    next_print = 0.0
    import time

    print("Esperando medidas. Ctrl+C para salir.", flush=True)
    while True:
        events = dict(poller.poll(500))
        if sock not in events:
            if latest is None:
                print("Esperando medidas...", flush=True)
            continue
        while True:
            try:
                raw = sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            msg = json.loads(raw)
            if msg.get("type") == "config":
                if not printed_config:
                    print(msg.get("formula", ""), flush=True)
                    printed_config = True
                continue
            if msg.get("type") == "sample":
                latest = msg
                if hz <= 0:
                    print(format_sample(msg), flush=True)
        if latest is None or hz <= 0:
            continue
        now = time.perf_counter()
        if now >= next_print:
            print(format_sample(latest), flush=True)
            next_print = now + 1.0 / hz


def _series(samples, path):
    import numpy as np

    rows = []
    for sample in samples:
        cursor = sample
        for key in path:
            cursor = cursor[key]
        rows.append(cursor)
    return np.asarray(rows, dtype=float)


def build_figure(show_truth: bool):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(11, 9), num="Sensores del cuadricóptero")
    titles = (
        "Acelerómetro (m/s²)",
        "Giroscopio (rad/s)",
        "Magnetómetro (µT)",
        "Corriente BLDC (A)",
        "Altímetro (m)",
    )
    meas_lines = []
    truth_lines = []
    for ax, title, n, colors, labels in (
        (axes[0], titles[0], 3, COLORS, ("x", "y", "z")),
        (axes[1], titles[1], 3, COLORS, ("x", "y", "z")),
        (axes[2], titles[2], 3, COLORS, ("x", "y", "z")),
        (axes[3], titles[3], 4, MOTOR_COLORS, MOTOR_LABELS),
        (axes[4], titles[4], 1, ("#4c4c4c",), ("alt",)),
    ):
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, alpha=0.35)
        ax.tick_params(labelsize=8)
        group_m = []
        group_t = []
        for i in range(n):
            (line,) = ax.plot([], [], color=colors[i], lw=1.15, label=labels[i])
            group_m.append(line)
            (ghost,) = ax.plot([], [], color=colors[i], lw=1.0, ls="--", alpha=0.75, label="_nolegend_")
            ghost.set_visible(show_truth)
            group_t.append(ghost)
        meas_lines.append(group_m)
        truth_lines.append(group_t)
        if n > 1:
            ax.legend(loc="upper right", fontsize=7, ncol=n, framealpha=0.85)
    axes[-1].set_xlabel("t (s)")
    proxies = [Line2D([0], [0], color="0.35", lw=1.15, label="medida")]
    if show_truth:
        proxies.append(Line2D([0], [0], color="0.35", lw=1.0, ls="--", label="verdad"))
    axes[0].legend(handles=list(axes[0].get_legend_handles_labels()[0])[:3] + proxies, loc="upper right", fontsize=7, ncol=5)
    fig.suptitle("Medidas del cuadricóptero    y = offset + scale·x + nonlin·x·|x| + ruido", fontsize=11)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    readout = fig.text(0.01, 0.008, "", fontsize=8, family="monospace")
    return fig, axes, meas_lines, truth_lines, readout


def refresh(axes, meas_lines, truth_lines, readout, samples, window: float, show_truth: bool, gaps: int) -> None:
    import numpy as np

    if not samples:
        readout.set_text("Esperando muestras...")
        return
    t = _series(samples, ("t",))
    t_end = float(t[-1])
    axes[-1].set_xlim(max(float(t[0]), t_end - window), t_end + 0.02 * window)

    channels = (
        ("meas", "accel"),
        ("meas", "gyro"),
        ("meas", "mag"),
        ("meas", "current"),
        ("meas", "altimeter"),
    )
    truth_channels = (
        ("truth", "accel"),
        ("truth", "gyro"),
        ("truth", "mag"),
        ("truth", "current"),
        ("truth", "altimeter"),
    )
    has_truth = show_truth and "truth" in samples[-1]
    for ax, lines, ghost, key, tkey in zip(axes, meas_lines, truth_lines, channels, truth_channels):
        values = _series(samples, key)
        if values.ndim == 1:
            values = values[:, None]
        truth = _series(samples, tkey) if has_truth else None
        if truth is not None and truth.ndim == 1:
            truth = truth[:, None]
        low = float(np.min(values))
        high = float(np.max(values))
        for i, line in enumerate(lines):
            line.set_data(t, values[:, i])
            if truth is not None:
                ghost[i].set_data(t, truth[:, i])
                low = min(low, float(np.min(truth[:, i])))
                high = max(high, float(np.max(truth[:, i])))
        pad = 0.08 * (high - low) + 0.05
        ax.set_ylim(low - pad, high + pad)

    last = samples[-1]["meas"]
    alt = last["altimeter"]
    cur = last["current"]
    readout.set_text(
        f"t={t_end:7.2f} s   alt={alt:7.3f} m   "
        f"I=[{cur[0]:5.2f} {cur[1]:5.2f} {cur[2]:5.2f} {cur[3]:5.2f}] A   "
        f"muestras perdidas={gaps}"
    )


def plot_loop(sock, window: float, show_truth: bool, endpoint: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, axes, meas_lines, truth_lines, readout = build_figure(show_truth)
    samples: deque = deque()
    state = {"gaps": 0, "last_seq": None, "config_seen": False}

    def drain(_frame):
        while True:
            try:
                raw = sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "config" and not state["config_seen"]:
                print(msg.get("formula", ""), flush=True)
                order = msg.get("motor_order")
                if order:
                    print("motores:", ", ".join(order), "  marco:", msg.get("frame_body", ""), flush=True)
                state["config_seen"] = True
                continue
            if kind != "sample":
                continue
            seq = msg.get("seq")
            if state["last_seq"] is not None and isinstance(seq, int) and seq > state["last_seq"] + 1:
                state["gaps"] += seq - state["last_seq"] - 1
            state["last_seq"] = seq
            samples.append(msg)
        if samples:
            t_end = samples[-1]["t"]
            while len(samples) > 2 and samples[0]["t"] < t_end - window:
                samples.popleft()
            while len(samples) > 20000:
                samples.popleft()
        refresh(axes, meas_lines, truth_lines, readout, samples, window, show_truth, state["gaps"])
        fig.canvas.draw_idle()
        return ()

    readout.set_text(f"Esperando muestras en {endpoint}")
    # cache_frame_data=False evita que matplotlib guarde cada frame.
    _anim = FuncAnimation(fig, drain, interval=80, cache_frame_data=False)
    plt.show()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ctx, sock = open_sub(args.connect)
    try:
        if args.text:
            text_loop(sock, args.hz)
        else:
            plot_loop(sock, args.window, not args.no_truth, args.connect)
    except KeyboardInterrupt:
        print()
    finally:
        sock.close(0)
        ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

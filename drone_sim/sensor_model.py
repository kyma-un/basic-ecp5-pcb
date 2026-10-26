"""Corrupción de sensores: offset, escala, no linealidad y ruido blanco."""

from __future__ import annotations

import numpy as np


FORMULA = "y = offset + scale * x + nonlin * x * |x| + N(0, noise)"


class AxisModel:
    """Un sensor de n ejes.

    ``noise`` es la desviación estándar de cada muestra (no densidad espectral).
    Para un Kalman discreto a la frecuencia de publicación, R = diag(noise²).
    La no linealidad es impar: ``nonlin * x * |x|``.
    """

    def __init__(self, noise, offset, scale, nonlin, n: int, name: str):
        self.name = name
        self.n = n
        self.noise = _as_vec(noise, n, name, "noise")
        self.offset = _as_vec(offset, n, name, "offset")
        self.scale = _as_vec(scale, n, name, "scale")
        self.nonlin = _as_vec(nonlin, n, name, "nonlin")
        if np.any(self.noise < 0):
            raise ValueError(f"{name}: el ruido debe ser >= 0")

    def apply(self, x, rng: np.random.Generator):
        raw = np.asarray(x, dtype=float).reshape(-1)
        if raw.shape != (self.n,):
            raise ValueError(f"{self.name}: se esperaban {self.n} valores y llegaron {raw.size}")
        y = self.offset + self.scale * raw + self.nonlin * raw * np.abs(raw)
        if np.any(self.noise > 0):
            y = y + rng.standard_normal(self.n) * self.noise
        if self.n == 1:
            return float(y[0])
        return y

    def to_dict(self) -> dict:
        return {
            "noise": self.noise.tolist(),
            "offset": self.offset.tolist(),
            "scale": self.scale.tolist(),
            "nonlin": self.nonlin.tolist(),
        }

    def summary(self) -> str:
        def fmt(v: np.ndarray) -> str:
            if v.size == 1:
                return f"{float(v[0]):.4g}"
            return "[" + ", ".join(f"{x:.4g}" for x in v) + "]"

        return (
            f"{self.name:8}  noise={fmt(self.noise):22}  offset={fmt(self.offset):22}  "
            f"scale={fmt(self.scale):22}  nonlin={fmt(self.nonlin)}"
        )


def _as_vec(value, n: int, name: str, field: str) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1 and n > 1:
        arr = np.full(n, float(arr[0]))
    if arr.shape != (n,):
        raise ValueError(f"{name}.{field}: se esperaban {n} valores y llegaron {arr.tolist()}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}.{field}: hay valores no finitos")
    return arr


def _check() -> None:
    rng = np.random.default_rng(0)
    model = AxisModel(
        noise=[0, 0, 0],
        offset=[1, 0, -1],
        scale=[2, 1, 1],
        nonlin=[0.5, 0, -0.25],
        n=3,
        name="t",
    )
    y = model.apply(np.array([2.0, -3.0, 4.0]), rng)
    # 1 + 2*2 + 0.5*2*2 = 7
    # 0 + (-3) = -3
    # -1 + 4 + (-0.25)*16 = -1
    assert np.allclose(y, [7.0, -3.0, -1.0]), y
    clean = AxisModel(0, 0, 1, 0, n=3, name="z")
    assert np.allclose(clean.apply(np.array([1.0, 2.0, 3.0]), rng), [1, 2, 3])
    alt = AxisModel(0.0, 0.5, 1.0, 0.0, n=1, name="alt")
    assert alt.apply(3.0, rng) == 3.5
    odd = AxisModel(0, 0, 1, -0.1, n=1, name="odd")
    # x*|x| conserva el signo: comprime ambos lados
    assert abs(odd.apply(2.0, rng) - 1.6) < 1e-12
    assert abs(odd.apply(-2.0, rng) - (-1.6)) < 1e-12


if __name__ == "__main__":
    _check()
    print("sensor_model ok")

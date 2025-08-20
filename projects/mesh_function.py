import numpy as np
from collections.abc import Callable


def mesh_function(f: Callable[[float], float], t: np.ndarray) -> np.ndarray:
    vf = np.vectorize(f)
    return vf(t)


def func(t: float) -> float:
    if t >= 0 and t <= 3:
        return np.exp(-t)
    elif t > 3 and t <= 4:
        return np.exp(-3*t)
    else:
        raise ValueError("Function only accepts argument t = [0, 4]")


def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)

if __name__ == "__main__":
    test_mesh_function()

import numpy as np

from boututils.fft_integrate import fft_integrate
from boututils.spectrogram import spectrogram


def test_fft_integrate():
    n = 64
    x = 2.0 * np.pi * np.arange(n) / n
    result = fft_integrate(np.cos(x))
    dx = x[1] - x[0]
    np.testing.assert_allclose(np.real(result) * dx, np.sin(x), atol=1e-10)


def test_int_func():
    from boututils.int_func import int_func

    x = np.linspace(0.0, 1.0, 101)
    result = int_func(x, 3.0 * x**2)
    np.testing.assert_allclose(result, x**3, atol=1e-6)


def test_spectrogram():
    t = np.linspace(0.0, 1.0, 256)
    data = np.sin(2.0 * np.pi * 20.0 * t)
    result = spectrogram(data, t[1] - t[0], 0.1, nskip=1)
    assert len(result) == 3

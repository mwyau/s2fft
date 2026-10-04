import numpy as np
import pytest
from jax import config

from s2fft.base_transforms import spherical
from s2fft.sampling import s2_samples as samples
from s2fft.utils import quadrature, quadrature_jax, quadrature_torch

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("L", [5, 6])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh", "gl"])
@pytest.mark.parametrize("method", ["numpy", "jax", "torch"])
def test_quadrature_mw_weights(flm_generator, L: int, sampling: str, method: str):
    spin = 0

    if method.lower() == "numpy":
        q = quadrature.quad_weights(L, sampling, spin)
    elif method.lower() == "jax":
        q = quadrature_jax.quad_weights(L, sampling)
    elif method.lower() == "torch":
        q = quadrature_torch.quad_weights(L, sampling).numpy()

    flm = flm_generator(L, spin, reality=False)

    f = spherical.inverse(flm, L, spin, sampling)

    integral = flm[0, 0 + L - 1] * np.sqrt(4 * np.pi)
    q = np.reshape(q, (-1, 1))

    nphi = samples.nphi_equiang(L, sampling)
    Q = q.dot(np.ones((1, nphi)))

    print(q.shape)
    print(Q.shape)
    print(f.shape)
    integral_check = np.sum(Q * f)

    np.testing.assert_allclose(integral, integral_check, atol=1e-14)


def test_quadrature_exceptions():
    L = 10

    with pytest.raises(ValueError):
        quadrature.quad_weights_transform(L, sampling="foo")

    with pytest.raises(ValueError):
        quadrature.quad_weights(L, sampling="foo")


@pytest.mark.parametrize("module", [quadrature, quadrature_jax, quadrature_torch])
@pytest.mark.parametrize("nphi", [12, 15])
def test_gl_physical_quadrature(module, nphi):
    L = 6
    q = np.asarray(module.quad_weights(L, "gl", nphi=nphi))
    np.testing.assert_allclose(nphi * q.sum(), 4 * np.pi, rtol=1e-14)
    np.testing.assert_array_equal(q, module.quad_weights_gl(L, nphi))
    default = module.quad_weights_gl(L)
    np.testing.assert_array_equal(default, module.quad_weights_gl(L, None))
    np.testing.assert_array_equal(default, module.quad_weights_gl(L, 2 * L - 1))
    np.testing.assert_array_equal(default, module.quad_weights(L, "gl"))
    np.testing.assert_array_equal(default, module.quad_weights(L, "gl", nphi=None))
    np.testing.assert_array_equal(default, module.quad_weights(L, "gl", nphi=2 * L - 1))
    np.testing.assert_array_equal(default, module.quad_weights_transform(L, "gl"))
    np.testing.assert_allclose(q * nphi, np.asarray(default) * (2 * L - 1), rtol=1e-14)


@pytest.mark.parametrize("module", [quadrature, quadrature_jax, quadrature_torch])
def test_gl_quadrature_invalid_longitude_counts(module):
    for nphi in [10, 0, -1, 12.5, True]:
        for weights in [module.quad_weights_gl, module.quad_weights]:
            kwargs = {"sampling": "gl"} if weights == module.quad_weights else {}
            with pytest.raises(ValueError, match="nphi"):
                weights(6, nphi=nphi, **kwargs)
    for sampling in ["mw", "mwss", "dh", "healpix"]:
        with pytest.raises(ValueError, match="only supported for GL"):
            module.quad_weights(6, sampling, nphi=12)

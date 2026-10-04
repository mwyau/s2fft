import numpy as np
import pytest

from s2fft.base_transforms import spherical
from s2fft.utils import resampling


def test_periodic_extension_invalid_sampling():
    f_dummy = np.zeros((2, 2), dtype=np.complex128)

    with pytest.raises(ValueError):
        resampling.periodic_extension(f_dummy, L=5, spin=0, sampling="healpix")

    with pytest.raises(ValueError):
        resampling.periodic_extension(f_dummy, L=5, spin=0, sampling="dh")


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize(
    "spin_reality", [(0, True), (0, False), (1, False), (2, False)]
)
def test_periodic_extension_mwss(flm_generator, L: int, spin_reality):
    (spin, reality) = spin_reality
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f = spherical.inverse(flm, L, spin, sampling="mwss")
    f = np.expand_dims(f, 0)

    f_ext = resampling.periodic_extension(f, L, spin, sampling="mwss")

    f_ext_check = resampling.periodic_extension_spatial_mwss(f, L, spin)

    np.testing.assert_allclose(f_ext, f_ext_check, atol=1e-10)


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize(
    "spin_reality", [(0, True), (0, False), (1, False), (2, False)]
)
def test_mwss_upsample_downsample(flm_generator, L: int, spin_reality):
    (spin, reality) = spin_reality
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f = spherical.inverse(flm, L, spin, sampling="mwss")
    f = np.expand_dims(f, 0)

    f_ext = resampling.periodic_extension_spatial_mwss(f, L, spin)

    f_ext_up = resampling.upsample_by_two_mwss_ext(f_ext, L)

    f_ext_up_down = resampling.downsample_by_two_mwss(f_ext_up, 2 * L)

    np.testing.assert_allclose(f_ext, f_ext_up_down, atol=1e-10)


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize("sampling", ["mw", "mwss"])
@pytest.mark.parametrize(
    "spin_reality", [(0, True), (0, False), (1, False), (2, False)]
)
def test_unextend(flm_generator, L: int, sampling: str, spin_reality):
    (spin, reality) = spin_reality
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f = spherical.inverse(flm, L, spin, sampling=sampling)
    f = np.expand_dims(f, 0)
    f_ext = resampling.periodic_extension(f, L, spin, sampling=sampling)

    f_unext = resampling.unextend(f_ext, L, sampling)

    np.testing.assert_allclose(f, f_unext, atol=1e-10)


def test_resampling_exceptions():
    f_dummy = np.zeros((2, 2), dtype=np.complex128)

    with pytest.raises(ValueError):
        L_odd = 3
        resampling.downsample_by_two_mwss(f_dummy, L_odd)

    with pytest.raises(ValueError):
        resampling.unextend(f_dummy, L=5, sampling="healpix")

    with pytest.raises(ValueError):
        resampling.unextend(f_dummy, L=5, sampling="dh")

    with pytest.raises(ValueError):
        # f_ext has wrong shape
        resampling.unextend(f_dummy, L=5, sampling="mw")

    with pytest.raises(ValueError):
        resampling.mw_to_mwss_phi(f_dummy, L=5)


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize(
    "spin_reality", [(0, True), (0, False), (1, False), (2, False)]
)
def test_mw_to_mwss_theta(flm_generator, L: int, spin_reality):
    (spin, reality) = spin_reality
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f_mw = spherical.inverse(flm, L, spin, sampling="mw")
    f_mwss = spherical.inverse(flm, L, spin, sampling="mwss")

    f_mwss_converted = resampling.mw_to_mwss(f_mw, L, spin)

    np.testing.assert_allclose(f_mwss, f_mwss_converted, atol=1e-10)


@pytest.mark.parametrize("L", [1, 5, 6])
@pytest.mark.parametrize("spin", [0, -1, 1, -2, 2])
@pytest.mark.parametrize("m_start", [-6, -5, 0])
def test_cc_resampling_adjoints(rng, L, spin, m_start):
    from s2fft.utils import resampling_jax

    x = rng.normal(size=(2, L + 1, 2 * L)) + 1j * rng.normal(size=(2, L + 1, 2 * L))
    y = rng.normal(size=(2, 2 * L, 2 * L)) + 1j * rng.normal(size=(2, 2 * L, 2 * L))
    for module in (resampling, resampling_jax):
        extended = module.periodic_extension_cc(x, spin, m_start)
        assert extended.shape == y.shape
        np.testing.assert_allclose(
            extended[..., L + 1 :, :],
            x[..., 1:-1, :][..., ::-1, :]
            * (-1.0) ** (np.arange(m_start, m_start + 2 * L) + spin),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.vdot(extended, y),
            np.vdot(x, module.periodic_extension_cc_adjoint(y, spin, m_start)),
            atol=1e-12,
        )
        shifted = module.half_grid_shift_cc(y)
        np.testing.assert_allclose(
            module.half_grid_shift_cc(shifted, adjoint=True),
            y
            - ((-1.0) ** np.arange(2 * L))[None, :, None]
            * np.fft.fft(y, axis=-2)[:, L : L + 1, :]
            / (2 * L),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.vdot(module.half_grid_shift_cc(extended), y),
            np.vdot(extended, module.half_grid_shift_cc(y, adjoint=True)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            shifted, resampling.half_grid_shift_cc(y), atol=1e-14
        )


@pytest.mark.parametrize("spin", [0, -1, 2])
def test_cc_folded_quadrature_hermitian(rng, spin):
    from s2fft.utils import resampling_jax

    L = 6
    x = rng.normal(size=(L + 1, 2 * L)) + 1j * rng.normal(size=(L + 1, 2 * L))
    y = rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)
    for module in (resampling, resampling_jax):
        ux = module.folded_cc_quadrature(x, L, spin, -L)
        uy = module.folded_cc_quadrature(y, L, spin, -L)
        np.testing.assert_allclose(np.vdot(x, uy), np.vdot(ux, y), atol=1e-13)
        np.testing.assert_allclose(
            ux, resampling.folded_cc_quadrature(x, L, spin, -L), atol=1e-14
        )


@pytest.mark.parametrize("spin", [0, -1, 1, -2, 2])
def test_cc_extension_matches_mwss_convention(rng, spin):
    L = 5
    f = rng.normal(size=(1, L + 1, 2 * L)) + 1j * rng.normal(size=(1, L + 1, 2 * L))
    ftm = np.fft.fftshift(np.fft.fft(f, axis=-1), axes=-1)
    extended = resampling.periodic_extension_cc(ftm, spin, m_start=-L)
    expected = resampling.periodic_extension_spatial_mwss(f, L, spin)
    expected = np.fft.fftshift(np.fft.fft(expected, axis=-1), axes=-1)
    np.testing.assert_allclose(extended, expected, atol=1e-14)


@pytest.mark.parametrize("mode", [-4, -1, 0, 1, 4])
def test_cc_half_grid_shift_modes(mode):
    from s2fft.utils import resampling_jax

    L = 5
    theta = np.arange(2 * L) * np.pi / L
    x = np.exp(1j * mode * theta)[:, None]
    expected = np.exp(1j * mode * (theta + np.pi / (2 * L)))[:, None]
    for module in (resampling, resampling_jax):
        np.testing.assert_allclose(module.half_grid_shift_cc(x), expected, atol=1e-14)


def test_cc_half_grid_shift_real_nyquist(rng):
    from s2fft.utils import resampling_jax

    L = 6
    x = rng.normal(size=(2 * L, 3))
    nyquist = np.broadcast_to((-1.0) ** np.arange(2 * L)[:, None], x.shape)
    for module in (resampling, resampling_jax):
        np.testing.assert_allclose(module.half_grid_shift_cc(nyquist), 0, atol=1e-14)
        np.testing.assert_allclose(
            np.asarray(module.half_grid_shift_cc(x)).imag, 0, atol=1e-14
        )


@pytest.mark.parametrize("L", [5, 6])
@pytest.mark.parametrize("spin", [0, -1, 2])
def test_cc_folded_quadrature_matches_dense_integral(L, spin):
    from s2fft.utils import quadrature, resampling_jax

    modes = np.arange(-L, L)
    even = (modes + spin) % 2 == 0

    def signal(theta, degree):
        return np.where(
            even, np.cos(degree * theta[:, None]), np.sin(degree * theta[:, None])
        ) * (1 + 0.3j)

    theta = np.arange(L + 1) * np.pi / L
    dense_theta = np.arange(2 * L + 1) * np.pi / (2 * L)
    x = signal(theta, L - 1)
    projection = signal(theta, L - 3)
    weights = quadrature.quad_weights_cc_theta_only(2 * L) * (np.pi / L)
    expected = np.sum(
        np.conj(signal(dense_theta, L - 3))
        * signal(dense_theta, L - 1)
        * weights[:, None],
        axis=0,
    )
    for module in (resampling, resampling_jax):
        actual = np.sum(
            np.conj(projection) * module.folded_cc_quadrature(x, L, spin, -L), axis=0
        )
        np.testing.assert_allclose(actual, expected, atol=1e-14)

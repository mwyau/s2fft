"""Full-bandwidth GL grids with physical longitude counts independent of ftm."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.test_util import check_grads

from s2fft.base_transforms import spherical as base
from s2fft.precompute_transforms import construct
from s2fft.precompute_transforms import spherical as precompute
from s2fft.recursions.price_mcewen import generate_precomputes
from s2fft.sampling import s2_samples as samples
from s2fft.transforms import spherical
from s2fft.utils import longitude

jax.config.update("jax_enable_x64", True)

# Match existing spherical tests: recursion precomputes contain unused singular entries.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

WIDTH_OFFSETS = [-1, 0, 1, 4]
FIELDS = [(0, True), (0, False), (-2, False), (1, False)]


def _array(value, method):
    return torch.from_numpy(value) if method == "torch" else value


def _numpy(value):
    return (
        value.detach().resolve_conj().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


def _longitude_reference(f, nphi):
    """Evaluate the native grid's trigonometric polynomial by an explicit sum."""
    L = f.shape[0]
    orders = np.arange(-L + 1, L)
    native_phis = 2 * np.pi * np.arange(2 * L - 1) / (2 * L - 1)
    modes = f @ np.exp(-1j * native_phis[:, None] * orders) / (2 * L - 1)
    phis = 2 * np.pi * np.arange(nphi) / nphi
    return modes @ np.exp(1j * orders[:, None] * phis)


@pytest.mark.parametrize("L", [6, 7])
@pytest.mark.parametrize("offset", WIDTH_OFFSETS)
@pytest.mark.parametrize("spin,reality", FIELDS)
@pytest.mark.parametrize("method", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("module", [spherical, precompute])
def test_gl_transform(flm_generator, L, offset, spin, reality, method, module):
    nphi = 2 * L + offset
    flm = flm_generator(L=L, spin=spin, reality=reality)
    kwargs = dict(L=L, spin=spin, sampling="gl", reality=reality, method=method)
    native = spherical.inverse(flm, **(kwargs | {"method": "numpy"}))
    expected = _longitude_reference(native, nphi)
    f = module.inverse(_array(flm, method), nphi=nphi, **kwargs)
    assert f.shape == (L, nphi)
    np.testing.assert_allclose(_numpy(f), expected, rtol=1e-12, atol=1e-12)
    recovered = module.forward(f, **kwargs)
    assert recovered.shape == (L, 2 * L - 1)
    np.testing.assert_allclose(_numpy(recovered), flm, rtol=1e-12, atol=1e-12)
    if offset == -1:
        # The explicit legacy width follows the same code as the omitted default.
        np.testing.assert_array_equal(
            _numpy(f), _numpy(module.inverse(_array(flm, method), **kwargs))
        )
    if nphi % 2 == 0:
        np.testing.assert_allclose(
            np.fft.fft(_numpy(f), axis=-1)[:, nphi // 2], 0, atol=1e-12
        )


@pytest.mark.parametrize("L", [6, 7])
@pytest.mark.parametrize("offset", WIDTH_OFFSETS)
@pytest.mark.parametrize("spin,reality", FIELDS)
@pytest.mark.parametrize("method", ["direct", "sov", "sov_fft", "sov_fft_vectorized"])
def test_gl_base(flm_generator, L, offset, spin, reality, method):
    flm = flm_generator(L=L, spin=spin, reality=reality)
    kwargs = dict(L=L, spin=spin, sampling="gl", reality=reality, method=method)
    f = base._inverse(flm, nphi=2 * L + offset, **kwargs)
    expected = _longitude_reference(
        base.inverse(flm, L, spin, "gl", reality=reality), f.shape[-1]
    )
    np.testing.assert_allclose(f, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(base._forward(f, **kwargs), flm, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("offset", WIDTH_OFFSETS)
@pytest.mark.parametrize("reality", [False, True])
@pytest.mark.parametrize("xp", [np, jnp])
def test_gl_fourier_orders(offset, reality, xp):
    L = 5
    nphi = 2 * L + offset
    phis = 2 * np.pi * np.arange(nphi) / nphi
    # Exercise both extreme orders and m=0 with distinct amplitudes.
    f = (
        (2 + 3j) * np.exp(1j * (L - 1) * phis)
        + (4 - 5j) * np.exp(-1j * (L - 1) * phis)
        + 7
    )
    if reality:
        f = f.real
    f = np.tile(f, (L, 1))
    ftm = longitude._forward_gl(xp.asarray(f), L, reality, xp)
    expected = np.zeros((L, 2 * L - 1), complex)
    expected[:, L - 1] = 7 * (2 * L - 1)
    expected[:, -1] = ((2 + 3j + 4 + 5j) / 2 if reality else 2 + 3j) * (2 * L - 1)
    if not reality:
        expected[:, 0] = (4 - 5j) * (2 * L - 1)
    np.testing.assert_allclose(ftm, expected, rtol=1e-14, atol=1e-13)
    reconstructed = longitude._inverse_gl(ftm / (2 * L - 1), L, nphi, reality, xp)
    np.testing.assert_allclose(reconstructed, f, rtol=1e-14, atol=1e-13)


@pytest.mark.parametrize("nphi", [10, 14, 15])
@pytest.mark.parametrize("reality", [True, False])
@pytest.mark.parametrize("module", [spherical, precompute, base])
def test_gl_discards_out_of_band_orders(nphi, reality, module):
    L = 5
    phis = 2 * np.pi * np.arange(nphi) / nphi
    order = nphi // 2  # Nyquist for even widths; above L-1 for odd widths.
    f = np.tile(np.exp(1j * order * phis), (L, 1))
    if reality:
        f = f.real
    kwargs = dict(L=L, sampling="gl", reality=reality)
    if module != base:
        kwargs["method"] = "numpy"
    np.testing.assert_allclose(module.forward(f, **kwargs), 0, atol=1e-13)


@pytest.fixture
def cached_ducc_gl_scalar_test_case(cached_test_case_wrapper, flm_generator):
    def generate_data(L, nphi, reality):
        ducc = pytest.importorskip("ducc0")
        flm = flm_generator(L=L, reality=reality)
        orders = np.arange(-L + 1, L)
        conjugate = (-1.0) ** orders * np.conj(flm[:, ::-1])
        # Split a complex scalar into two real fields in DUCC's packed convention.
        components = [(flm + conjugate) / 2, (flm - conjugate) / (2j)]
        maps, coefficients = [], []
        for component in components:
            alm = samples.flm_2d_to_hp(component, L)[None, :]
            field = ducc.sht.experimental.synthesis_2d(
                alm=alm, spin=0, lmax=L - 1, geometry="GL", ntheta=L, nphi=nphi
            )[0]
            maps.append(field)
            analyzed = ducc.sht.experimental.analysis_2d(
                map=field[None, :], spin=0, lmax=L - 1, geometry="GL"
            )[0]
            coefficients.append(samples.flm_hp_to_2d(analyzed, L))
        return {
            "flm": flm,
            "f_ducc": maps[0] + 1j * maps[1],
            "flm_ducc": coefficients[0] + 1j * coefficients[1],
        }

    return cached_test_case_wrapper(generate_data, "npz")


@pytest.mark.parametrize("L", [6, 7])
@pytest.mark.parametrize("offset", WIDTH_OFFSETS)
@pytest.mark.parametrize("reality", [True, False])
def test_gl_ducc_scalar(cached_ducc_gl_scalar_test_case, L, offset, reality):
    data = cached_ducc_gl_scalar_test_case(L, 2 * L + offset, reality)
    flm, expected = data["flm"], data["f_ducc"]
    f = spherical.inverse(flm, L, sampling="gl", reality=reality, nphi=2 * L + offset)
    np.testing.assert_allclose(f, expected, rtol=1e-12, atol=1e-12)
    recovered = spherical.forward(
        expected.real if reality else expected, L, sampling="gl", reality=reality
    )
    np.testing.assert_allclose(recovered, flm, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(data["flm_ducc"], flm, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("module", [spherical, precompute, base])
@pytest.mark.parametrize("reality", [True, False])
def test_gl_iterative_refinement(flm_generator, module, reality):
    L, nphi = 6, 16
    flm = flm_generator(L=L, reality=reality)
    kwargs = dict(L=L, sampling="gl", reality=reality)
    if module != base:
        kwargs["method"] = "jax"
    f = module.inverse(flm, nphi=nphi, **kwargs)
    np.testing.assert_allclose(
        module.forward(f, iter=1, **kwargs), flm, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("method", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("reality", [True, False])
def test_gl_reuses_precomputes(flm_generator, method, reality):
    L = 7
    flm = flm_generator(L=L, reality=reality, L_lower=2)
    kwargs = dict(L=L, sampling="gl", reality=reality, method=method, L_lower=2)
    inverse_precomps = generate_precomputes(L, sampling="gl", forward=False, L_lower=2)
    forward_precomps = generate_precomputes(L, sampling="gl", forward=True, L_lower=2)
    f = spherical.inverse(
        _array(flm, method), nphi=18, precomps=inverse_precomps, **kwargs
    )
    recovered = spherical.forward(f, precomps=forward_precomps, **kwargs)
    np.testing.assert_allclose(_numpy(recovered), flm, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("module", [spherical, precompute])
@pytest.mark.parametrize("offset", WIDTH_OFFSETS)
@pytest.mark.parametrize("spin,reality", [(0, True), (0, False), (1, False)])
def test_gl_jax_transformations(flm_generator, module, offset, spin, reality):
    L = 5
    flm = jnp.asarray(flm_generator(L=L, spin=spin, reality=reality))
    kwargs = dict(L=L, sampling="gl", reality=reality, spin=spin, method="jax")
    if module == precompute:
        inverse_kernel = construct.spin_spherical_kernel_jax(
            L, spin, reality, "gl", forward=False
        )
        forward_kernel = construct.spin_spherical_kernel_jax(
            L, spin, reality, "gl", forward=True
        )
        inv = partial(
            module.inverse, kernel=inverse_kernel, nphi=2 * L + offset, **kwargs
        )
        fwd = partial(module.forward, kernel=forward_kernel, **kwargs)
    else:
        inv = partial(module.inverse, nphi=2 * L + offset, **kwargs)
        fwd = partial(module.forward, **kwargs)
    f = jax.jit(inv)(flm)
    np.testing.assert_allclose(jax.jit(fwd)(f), flm, rtol=1e-12, atol=1e-12)
    batch = jnp.stack([flm, 2 * flm])
    fields = jax.jit(jax.vmap(inv))(batch)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(fwd))(fields), batch, rtol=1e-12, atol=1e-12
    )
    for transform, primal in [(inv, flm), (fwd, f)]:
        tangent = primal * 0.3
        _, derivative = jax.jvp(transform, (primal,), (tangent,))
        np.testing.assert_allclose(
            derivative, transform(tangent), rtol=1e-12, atol=1e-12
        )

        def loss(x):
            return jnp.sum(jnp.abs(transform(x)) ** 2)

        check_grads(loss, (primal,), order=1, modes=("rev",), atol=1e-5, rtol=1e-5)
        batched_grad = jax.jit(jax.vmap(jax.grad(loss)))(
            jnp.stack([primal, 2 * primal])
        )
        np.testing.assert_allclose(
            batched_grad[1], 2 * batched_grad[0], rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("module", [spherical, precompute])
@pytest.mark.parametrize("reality", [True, False])
def test_gl_torch_gradients(flm_generator, module, reality):
    L = 6
    coefficients = torch.from_numpy(
        flm_generator(L=L, reality=reality)
    ).requires_grad_()
    kwargs = dict(L=L, sampling="gl", reality=reality, method="torch")
    f = module.inverse(coefficients, nphi=12, **kwargs)
    recovered = module.forward(f, **kwargs)
    loss = torch.sum(torch.abs(recovered) ** 2)
    loss.backward()
    expected_grad = 2 * _numpy(coefficients)
    if reality:
        # Synthesis uses m >= 0; the recovered negative orders depend on them.
        expected_grad[:, : L - 1] = 0
        expected_grad[:, L:] *= 2
        expected_grad[:, L - 1] = expected_grad[:, L - 1].real
    np.testing.assert_allclose(
        _numpy(coefficients.grad), expected_grad, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("module", [spherical, precompute, base])
@pytest.mark.parametrize("method", ["numpy", "jax", "torch"])
def test_gl_invalid_counts(module, method):
    if module == base and method != "numpy":
        pytest.skip("Base transforms use NumPy.")
    L = 6
    kwargs = dict(L=L, sampling="gl")
    if module != base:
        kwargs["method"] = method
    for nphi in [10, 0, -1, 12.5, True]:
        with pytest.raises(ValueError, match="nphi"):
            module.inverse(
                _array(np.zeros((L, 2 * L - 1), complex), method), nphi=nphi, **kwargs
            )
    with pytest.raises(ValueError, match="nphi"):
        module.forward(_array(np.zeros((L, 10)), method), **kwargs)
    with pytest.raises(ValueError, match="shape"):
        module.forward(_array(np.zeros((L + 1, 12)), method), **kwargs)
    for sampling in ["mw", "mwss", "dh", "healpix"]:
        with pytest.raises(ValueError, match="only supported for GL"):
            module.inverse(
                _array(np.zeros((L, 2 * L - 1), complex), method),
                nphi=12,
                **(kwargs | {"sampling": sampling}),
            )


def test_gl_ssht_custom_count_rejected():
    with pytest.raises(ValueError, match="jax_ssht"):
        spherical.inverse(
            np.zeros((6, 11)), 6, sampling="gl", method="jax_ssht", nphi=12
        )
    with pytest.raises(ValueError, match="jax_ssht"):
        spherical.forward(np.zeros((6, 12)), 6, sampling="gl", method="jax_ssht")


@pytest.mark.parametrize("L", [6, 7])
@pytest.mark.parametrize("offset", WIDTH_OFFSETS)
@pytest.mark.parametrize("spin", [-2, 1])
def test_gl_spin_ssht(cached_ssht_test_case, L, offset, spin):
    # SSHT uses the legacy GL grid; explicitly evaluate its longitude polynomial.
    data = cached_ssht_test_case(L, 0, spin, "gl", False)
    expected = _longitude_reference(data["f_ssht"], 2 * L + offset)
    f = spherical.inverse(data["flm"], L, spin, sampling="gl", nphi=2 * L + offset)
    np.testing.assert_allclose(f, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        spherical.forward(expected, L, spin, sampling="gl"),
        data["flm"],
        rtol=1e-12,
        atol=1e-12,
    )

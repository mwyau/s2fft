from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from s2fft.precompute_transforms import spherical as precompute
from s2fft.recursions.price_mcewen import generate_precomputes
from s2fft.sampling import s2_samples as samples
from s2fft.transforms import spherical

jax.config.update("jax_enable_x64", True)

L_to_test = [6, 7]
L_lower_to_test = [0, 2]
spin_to_test = [-2, 0, 1]
nside_to_test = [4, 5]
sampling_to_test = ["mw", "mwss", "dh", "gl"]
method_to_test = ["numpy", "jax", "torch"]
reality_to_test = [False, True]
multiple_gpus = [False, True]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.parametrize("use_generate_precomputes", [True, False])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_inverse(
    cached_ssht_test_case: Callable,
    L: int,
    L_lower: int,
    spin: int,
    sampling: str,
    method: str,
    reality: bool,
    spmd: bool,
    use_generate_precomputes: bool,
):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")
    if spmd and method != "jax":
        pytest.skip("GPU distribution only valid for JAX.")

    test_data = cached_ssht_test_case(L, L_lower, spin, sampling, reality)

    if use_generate_precomputes:
        precomps = generate_precomputes(L, spin, sampling, L_lower=L_lower)
    else:
        precomps = None
    f = spherical.inverse(
        torch.from_numpy(test_data["flm"]) if method == "torch" else test_data["flm"],
        L,
        spin,
        sampling=sampling,
        method=method,
        reality=reality,
        precomps=precomps,
        spmd=spmd,
        L_lower=L_lower,
    )
    np.testing.assert_allclose(f, test_data["f_ssht"], atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_inverse_healpix(
    cached_healpy_test_case: Callable,
    nside: int,
    method: str,
    spmd: bool,
):
    sampling = "healpix"
    L = 2 * nside
    reality = True
    test_data = cached_healpy_test_case(L=L, nside=nside, reality=reality)
    precomps = generate_precomputes(L, 0, sampling, nside, False)
    f = spherical.inverse(
        torch.from_numpy(test_data["flm"]) if method == "torch" else test_data["flm"],
        L,
        spin=0,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=reality,
        precomps=precomps,
        spmd=spmd,
    )

    np.testing.assert_allclose(np.real(f), np.real(test_data["f_hp"]), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.parametrize("use_generate_precomputes", [True, False])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_forward(
    cached_ssht_test_case: Callable,
    L: int,
    L_lower: int,
    spin: int,
    sampling: str,
    method: str,
    reality: bool,
    spmd: bool,
    use_generate_precomputes: bool,
):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")
    if spmd and method != "jax":
        pytest.skip("GPU distribution only valid for JAX.")

    test_data = cached_ssht_test_case(L, L_lower, spin, sampling, reality)

    if use_generate_precomputes:
        precomps = generate_precomputes(L, spin, sampling, None, True, L_lower)
    else:
        precomps = None

    flm_check = spherical.forward(
        torch.from_numpy(test_data["f_ssht"])
        if method == "torch"
        else test_data["f_ssht"],
        L,
        spin,
        sampling=sampling,
        method=method,
        reality=reality,
        precomps=precomps,
        spmd=spmd,
        L_lower=L_lower,
    )
    np.testing.assert_allclose(test_data["flm"], flm_check, atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.parametrize("iter", [0, 1, 2, 3])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_forward_healpix(
    cached_healpy_test_case: Callable,
    nside: int,
    method: str,
    spmd: bool,
    iter: int,
):
    sampling = "healpix"
    L = 2 * nside
    reality = True
    test_data = cached_healpy_test_case(L=L, nside=nside, reality=reality, n_iter=iter)

    precomps = generate_precomputes(L, 0, sampling, nside, True)
    flm_check = spherical.forward(
        torch.from_numpy(test_data["f_hp"]) if method == "torch" else test_data["f_hp"],
        L,
        spin=0,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=True,
        precomps=precomps,
        spmd=spmd,
        iter=iter,
    )
    flm_check = samples.flm_2d_to_hp(flm_check, L)

    np.testing.assert_allclose(test_data["flm_hp"], flm_check, atol=1e-14)


def test_spin_exceptions(flm_generator):
    spin = 10
    L = 16
    sampling = "mw"

    flm = flm_generator(L=L, reality=False)
    f = spherical.inverse(flm, L, spin=0, sampling=sampling, method="jax")

    with pytest.raises(Warning):
        spherical.inverse(flm, L, spin=spin, sampling=sampling, method="jax")

    with pytest.raises(Warning):
        spherical.forward(f, L, spin=spin, sampling=sampling, method="jax")


@pytest.mark.healpy
@pytest.mark.pyssht
def test_sampling_ssht_backend_exceptions(flm_generator):
    sampling = "healpix"
    nside = 6
    L = 2 * nside

    flm = flm_generator(L=L, reality=False)
    f = spherical.inverse(flm, L, 0, nside, sampling, "jax_healpy")

    with pytest.raises(ValueError):
        spherical.inverse(flm, L, 0, nside, sampling, "jax_ssht")

    with pytest.raises(ValueError):
        spherical.forward(f, L, 0, nside, sampling, "jax_ssht")


@pytest.mark.healpy
@pytest.mark.pyssht
def test_sampling_healpy_backend_exceptions(flm_generator):
    sampling = "mw"
    L = 12

    flm = flm_generator(L=L, reality=False)
    f = spherical.inverse(flm, L, 0, None, sampling, "jax_ssht")

    with pytest.raises(ValueError):
        spherical.inverse(flm, L, 0, None, sampling, "jax_healpy")

    with pytest.raises(ValueError):
        spherical.forward(f, L, 0, None, sampling, "jax_healpy")


def test_sampling_exceptions(flm_generator):
    with pytest.raises(ValueError):
        spherical.inverse(None, 0, 0, None, method="incorrect")

    with pytest.raises(ValueError):
        spherical.forward(None, 0, 0, None, method="incorrect")


@pytest.mark.parametrize("module", [spherical, precompute])
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", [False, True])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_gl_custom_width_roundtrip(flm_generator, module, method, reality):
    L = 6
    flm = flm_generator(L=L, reality=reality)
    coefficients = torch.from_numpy(flm) if method == "torch" else flm
    if method == "torch" and module == spherical and not reality:
        coefficients.requires_grad_()
    kwargs = dict(L=L, sampling="gl", method=method, reality=reality)
    native = module.inverse(coefficients, **kwargs)
    explicit = module.inverse(coefficients, nphi=2 * L - 1, **kwargs)
    f = module.inverse(coefficients, nphi=2 * L, **kwargs)
    recovered = module.forward(f, **kwargs)
    if method == "torch":
        if coefficients.requires_grad:
            torch.sum(torch.abs(recovered) ** 2).backward()
            np.testing.assert_allclose(
                coefficients.grad.resolve_conj().numpy(),
                2 * flm,
                rtol=1e-12,
                atol=1e-12,
            )
        native, explicit, f, recovered = (
            x.detach().resolve_conj().numpy() for x in [native, explicit, f, recovered]
        )
    np.testing.assert_array_equal(native, explicit)
    assert f.shape == (L, 2 * L)
    assert recovered.shape == (L, 2 * L - 1)
    np.testing.assert_allclose(recovered, flm, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("reality", [False, True])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_gl_nyquist_and_extra_modes(flm_generator, reality):
    L, nphi = 5, 14
    kwargs = dict(L=L, sampling="gl", method="numpy", reality=reality)
    flm = flm_generator(L=L, reality=reality)
    f = spherical.inverse(flm, nphi=nphi, **kwargs)
    np.testing.assert_allclose(
        np.fft.fft(f, axis=-1)[:, L : nphi - L + 1], 0, atol=1e-12
    )
    np.testing.assert_allclose(
        spherical.forward(f, **kwargs), flm, rtol=1e-12, atol=1e-12
    )
    phis = samples.phis_equiang(L, "gl", nphi)
    for order in [L, nphi // 2]:
        extra_mode = np.tile(np.exp(1j * order * phis), (L, 1))
        np.testing.assert_allclose(
            spherical.forward(extra_mode.real if reality else extra_mode, **kwargs),
            0,
            atol=1e-13,
        )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_gl_custom_width_jax_transformability(flm_generator):
    L = 5
    flm = jnp.asarray(flm_generator(L=L))
    kwargs = dict(L=L, sampling="gl", method="jax")
    inv = jax.jit(partial(spherical.inverse, nphi=2 * L, **kwargs))
    fwd = jax.jit(partial(spherical.forward, **kwargs))
    f, df = jax.jvp(inv, (flm,), (0.3 * flm,))
    recovered, dflm = jax.jvp(fwd, (f,), (df,))
    np.testing.assert_allclose(recovered, flm, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dflm, 0.3 * flm, rtol=1e-12, atol=1e-12)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_gl_custom_width_iterative_refinement(flm_generator):
    L = 6
    flm = flm_generator(L=L)
    kwargs = dict(L=L, sampling="gl", method="jax")
    f = spherical.inverse(flm, nphi=15, **kwargs)
    np.testing.assert_allclose(
        spherical.forward(f, iter=1, **kwargs), flm, rtol=1e-12, atol=1e-12
    )


def test_gl_ssht_custom_count_rejected():
    with pytest.raises(ValueError, match="jax_ssht"):
        spherical.inverse(
            np.zeros((6, 11)), 6, sampling="gl", method="jax_ssht", nphi=12
        )
    with pytest.raises(ValueError, match="jax_ssht"):
        spherical.forward(np.zeros((6, 12)), 6, sampling="gl", method="jax_ssht")


@pytest.fixture
def cached_ducc_gl_test_case(cached_test_case_wrapper, flm_generator):
    def generate_data(L, nphi):
        ducc = pytest.importorskip("ducc0")
        flm = flm_generator(L=L, reality=True)
        alm = samples.flm_2d_to_hp(flm, L)[None, :]
        f = ducc.sht.experimental.synthesis_2d(
            alm=alm, spin=0, lmax=L - 1, geometry="GL", ntheta=L, nphi=nphi
        )[0]
        return {"flm": flm, "f": f}

    return cached_test_case_wrapper(generate_data, "npz")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_gl_custom_width_ducc(cached_ducc_gl_test_case):
    L = 6
    data = cached_ducc_gl_test_case(L, 2 * L)
    kwargs = dict(L=L, sampling="gl", method="numpy", reality=True)
    f = spherical.inverse(data["flm"], nphi=2 * L, **kwargs)
    np.testing.assert_allclose(f, data["f"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        spherical.forward(data["f"], **kwargs), data["flm"], rtol=1e-12, atol=1e-12
    )

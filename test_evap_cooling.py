import mpmath as mp

from evap_cooling import BoxTrapModel, HarmonicTrapModel, QuadrupoleTrapModel


def test_harmonic_fermion_mu():
    model = HarmonicTrapModel(omega=2 * mp.pi * 100)
    x = model.solve_chemical_potential(
        n_atoms=1e7,
        temperature=5e-5,
        boson=False,
        guess=-11.8,
        bracket=(-13, -10),
    )
    assert -13 < x < -10


def test_box_fermion_mu():
    model = BoxTrapModel(volume=5.84333746920601e-9)
    x = model.solve_chemical_potential(
        n_atoms=1e7,
        temperature=5e-5,
        boson=False,
        guess=-6,
        bracket=(-12, 2),
    )
    assert -12 < x < 2


def test_quadrupole_state_solver_smoke():
    model = QuadrupoleTrapModel(volume=9.20515692e-8 * 0.000002)
    t0 = 5e-5
    mu0 = -5 * model.c.kb * t0
    n0 = model.number(t0, mu0, boson=False)
    e0 = model.energy(n0, t0, mu0, boson=False)

    state = model.solve_state(
        n_target=float(n0),
        e_target=float(e0),
        boson=False,
        t_guess=4.8e-5,
        mu_guess=float(-4.6 * model.c.kb * t0),
    )

    assert abs(state.temperature - t0) / t0 < 1e-5

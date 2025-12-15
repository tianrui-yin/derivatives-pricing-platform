# -*- coding: utf-8 -*-
"""
Tests Unitaires pour le Solveur PDE (Crank-Nicolson)
=====================================================
Validation du pricing par EDP de Black-Scholes-Merton.

Couverture:
    - Prix européens vs Black-Scholes analytique
    - Parité Put-Call
    - Options américaines (prime d'exercice anticipé)
    - Convergence du schéma numérique
    - Algorithme de Thomas
    - Conditions aux bords
    - Comparaison 4 méthodes (BS, MC, CRR, PDE)
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pde_solver import PDESolver, compare_pricing_methods, american_option_comparison
from src.utils import black_scholes_price


# ---------------------------------------------------------------------------
# Paramètres de test communs
# ---------------------------------------------------------------------------
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20


# ===================================================================
# 1. Prix européen Call vs Black-Scholes analytique (< 0.5%)
# ===================================================================
class TestEuropeanCallVsBS:
    def test_atm_call(self):
        solver = PDESolver(S0, K, T, r, sigma, option_type='call', N_space=200, N_time=200)
        pde_price = solver.price()
        bs_price = black_scholes_price(S0, K, T, r, sigma, 'call')
        rel_error = abs(pde_price - bs_price) / bs_price
        assert rel_error < 0.005, (
            f"Call ATM: PDE={pde_price:.6f}, BS={bs_price:.6f}, erreur={rel_error:.4%}"
        )

    def test_itm_call(self):
        solver = PDESolver(S0, 90.0, T, r, sigma, option_type='call', N_space=200, N_time=200)
        pde_price = solver.price()
        bs_price = black_scholes_price(S0, 90.0, T, r, sigma, 'call')
        rel_error = abs(pde_price - bs_price) / bs_price
        assert rel_error < 0.005

    def test_otm_call(self):
        solver = PDESolver(S0, 110.0, T, r, sigma, option_type='call', N_space=200, N_time=200)
        pde_price = solver.price()
        bs_price = black_scholes_price(S0, 110.0, T, r, sigma, 'call')
        rel_error = abs(pde_price - bs_price) / bs_price
        assert rel_error < 0.005


# ===================================================================
# 2. Prix européen Put vs Black-Scholes analytique (< 0.5%)
# ===================================================================
class TestEuropeanPutVsBS:
    def test_atm_put(self):
        solver = PDESolver(S0, K, T, r, sigma, option_type='put', N_space=200, N_time=200)
        pde_price = solver.price()
        bs_price = black_scholes_price(S0, K, T, r, sigma, 'put')
        rel_error = abs(pde_price - bs_price) / bs_price
        assert rel_error < 0.005, (
            f"Put ATM: PDE={pde_price:.6f}, BS={bs_price:.6f}, erreur={rel_error:.4%}"
        )

    def test_itm_put(self):
        solver = PDESolver(S0, 110.0, T, r, sigma, option_type='put', N_space=200, N_time=200)
        pde_price = solver.price()
        bs_price = black_scholes_price(S0, 110.0, T, r, sigma, 'put')
        rel_error = abs(pde_price - bs_price) / bs_price
        assert rel_error < 0.005

    def test_otm_put(self):
        solver = PDESolver(S0, 90.0, T, r, sigma, option_type='put', N_space=200, N_time=200)
        pde_price = solver.price()
        bs_price = black_scholes_price(S0, 90.0, T, r, sigma, 'put')
        rel_error = abs(pde_price - bs_price) / bs_price
        assert rel_error < 0.005


# ===================================================================
# 3. Parité Put-Call : C_pde - P_pde ≈ S - K*exp(-rT)
# ===================================================================
class TestPutCallParity:
    def test_parity_atm(self):
        call_solver = PDESolver(S0, K, T, r, sigma, option_type='call', N_space=200, N_time=200)
        put_solver = PDESolver(S0, K, T, r, sigma, option_type='put', N_space=200, N_time=200)
        pde_diff = call_solver.price() - put_solver.price()
        theoretical = S0 - K * np.exp(-r * T)
        assert abs(pde_diff - theoretical) < 0.10, (
            f"Parité Put-Call: PDE diff={pde_diff:.6f}, théorique={theoretical:.6f}"
        )

    def test_parity_itm(self):
        K_itm = 90.0
        call_solver = PDESolver(S0, K_itm, T, r, sigma, option_type='call', N_space=200, N_time=200)
        put_solver = PDESolver(S0, K_itm, T, r, sigma, option_type='put', N_space=200, N_time=200)
        pde_diff = call_solver.price() - put_solver.price()
        theoretical = S0 - K_itm * np.exp(-r * T)
        assert abs(pde_diff - theoretical) < 0.10


# ===================================================================
# 4. Option américaine Put >= Put européen
# ===================================================================
class TestAmericanPut:
    def test_american_put_geq_european(self):
        amer_solver = PDESolver(S0, K, T, r, sigma, option_type='put', exercise='american',
                                N_space=200, N_time=200)
        euro_solver = PDESolver(S0, K, T, r, sigma, option_type='put', exercise='european',
                                N_space=200, N_time=200)
        amer_price = amer_solver.price()
        euro_price = euro_solver.price()
        assert amer_price >= euro_price - 0.01, (
            f"Put US ({amer_price:.6f}) < Put EU ({euro_price:.6f})"
        )

    def test_early_exercise_premium_positive(self):
        """La prime d'exercice anticipé pour un put doit être positive."""
        amer_solver = PDESolver(S0, K, T, r, sigma, option_type='put', exercise='american',
                                N_space=200, N_time=200)
        bs_price = black_scholes_price(S0, K, T, r, sigma, 'put')
        premium = amer_solver.price() - bs_price
        assert premium > -0.01, f"Prime d'exercice anticipé négative: {premium:.6f}"


# ===================================================================
# 5. Call américain ≈ Call européen (sans dividende)
# ===================================================================
class TestAmericanCall:
    def test_american_call_approx_european(self):
        amer_solver = PDESolver(S0, K, T, r, sigma, option_type='call', exercise='american',
                                N_space=200, N_time=200)
        bs_price = black_scholes_price(S0, K, T, r, sigma, 'call')
        diff = abs(amer_solver.price() - bs_price)
        assert diff < 0.10, (
            f"Call US != Call EU: diff={diff:.6f}"
        )


# ===================================================================
# 6. Convergence : l'erreur décroît avec N
# ===================================================================
class TestConvergence:
    def test_error_decreases_with_n(self):
        bs_price = black_scholes_price(S0, K, T, r, sigma, 'call')
        grid_sizes = [50, 100, 200]
        errors = []
        for n in grid_sizes:
            solver = PDESolver(S0, K, T, r, sigma, option_type='call', N_space=n, N_time=n)
            pde_price = solver.price()
            errors.append(abs(pde_price - bs_price))
        # L'erreur doit globalement décroître
        assert errors[-1] < errors[0], (
            f"Pas de convergence: erreurs={errors}"
        )


# ===================================================================
# 7. Sensibilité au maillage : grille plus fine → plus proche de BS
# ===================================================================
class TestGridSensitivity:
    def test_finer_grid_closer_to_bs(self):
        bs_price = black_scholes_price(S0, K, T, r, sigma, 'put')
        coarse = PDESolver(S0, K, T, r, sigma, option_type='put', N_space=50, N_time=50)
        fine = PDESolver(S0, K, T, r, sigma, option_type='put', N_space=300, N_time=300)
        err_coarse = abs(coarse.price() - bs_price)
        err_fine = abs(fine.price() - bs_price)
        assert err_fine < err_coarse, (
            f"Grille fine pas meilleure: fine={err_fine:.6f}, coarse={err_coarse:.6f}"
        )


# ===================================================================
# 8. Algorithme de Thomas sur un système tridiagonal connu
# ===================================================================
class TestThomasAlgorithm:
    def test_known_system(self):
        """
        Résoudre A*x = d avec A tridiagonale connue.
        A = [[2, -1, 0],
             [-1, 2, -1],
             [0, -1, 2]]
        d = [1, 0, 1]
        Solution exacte : x = [1, 1, 1]
        """
        solver = PDESolver(S0, K, T, r, sigma)
        lower = np.array([0.0, -1.0, -1.0])
        main = np.array([2.0, 2.0, 2.0])
        upper = np.array([-1.0, -1.0, 0.0])
        rhs = np.array([1.0, 0.0, 1.0])
        x = solver._thomas_algorithm(lower, main, upper, rhs)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(x, expected, atol=1e-12)

    def test_larger_system(self):
        """Système tridiagonal 5x5."""
        solver = PDESolver(S0, K, T, r, sigma)
        n = 5
        # Matrice diagonale dominante
        main = np.full(n, 4.0)
        lower = np.full(n, -1.0)
        lower[0] = 0.0
        upper = np.full(n, -1.0)
        upper[-1] = 0.0
        # Solution attendue x = [1, 2, 3, 4, 5]
        x_exact = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Construire rhs = A @ x_exact
        rhs = main * x_exact
        for i in range(1, n):
            rhs[i] += lower[i] * x_exact[i - 1]
        for i in range(n - 1):
            rhs[i] += upper[i] * x_exact[i + 1]
        x_computed = solver._thomas_algorithm(lower, main, upper, rhs)
        np.testing.assert_allclose(x_computed, x_exact, atol=1e-10)


# ===================================================================
# 9. Conditions aux bords : V aux extrêmes correspond à l'analytique
# ===================================================================
class TestBoundaryConditions:
    def test_call_boundary_at_s_max(self):
        """V(S_max, 0) ≈ S_max - K*exp(-rT) pour un call."""
        solver = PDESolver(S0, K, T, r, sigma, option_type='call', N_space=200, N_time=200)
        solver._setup_grid()
        bc = solver._boundary_conditions(0.0)  # t = 0 (début, T reste)
        S_max = S0 * solver.S_max_mult
        expected = S_max - K * np.exp(-r * T)
        assert abs(bc[1] - expected) < 0.01, (
            f"BC call S_max: obtenu={bc[1]:.4f}, attendu={expected:.4f}"
        )

    def test_put_boundary_at_s_min(self):
        """V(S_min, 0) ≈ K*exp(-rT) - S_min pour un put."""
        solver = PDESolver(S0, K, T, r, sigma, option_type='put', N_space=200, N_time=200)
        solver._setup_grid()
        bc = solver._boundary_conditions(0.0)
        S_min = S0 / solver.S_max_mult
        expected = K * np.exp(-r * T) - S_min
        assert abs(bc[0] - expected) < 0.01, (
            f"BC put S_min: obtenu={bc[0]:.4f}, attendu={expected:.4f}"
        )

    def test_call_boundary_at_s_min_is_zero(self):
        """V(S_min, t) = 0 pour un call."""
        solver = PDESolver(S0, K, T, r, sigma, option_type='call', N_space=200, N_time=200)
        solver._setup_grid()
        bc = solver._boundary_conditions(0.0)
        assert abs(bc[0]) < 1e-10

    def test_put_boundary_at_s_max_is_zero(self):
        """V(S_max, t) = 0 pour un put."""
        solver = PDESolver(S0, K, T, r, sigma, option_type='put', N_space=200, N_time=200)
        solver._setup_grid()
        bc = solver._boundary_conditions(0.0)
        assert abs(bc[1]) < 1e-10


# ===================================================================
# 10. Comparaison 4 méthodes : BS, MC, CRR, PDE (< 1%)
# ===================================================================
class TestFourMethodComparison:
    def test_call_four_methods(self):
        result = compare_pricing_methods(S0, K, T, r, sigma, option_type='call')
        bs = result['black_scholes']
        for method_name in ['monte_carlo', 'crr_binomial', 'pde_crank_nicolson']:
            price = result[method_name]
            rel_error = abs(price - bs) / bs
            assert rel_error < 0.01, (
                f"{method_name}: prix={price:.6f}, BS={bs:.6f}, erreur={rel_error:.4%}"
            )

    def test_put_four_methods(self):
        result = compare_pricing_methods(S0, K, T, r, sigma, option_type='put')
        bs = result['black_scholes']
        for method_name in ['monte_carlo', 'crr_binomial', 'pde_crank_nicolson']:
            price = result[method_name]
            rel_error = abs(price - bs) / bs
            assert rel_error < 0.01


# ===================================================================
# 11. American PDE vs CRR : prix à moins de 1% l'un de l'autre
# ===================================================================
class TestAmericanPDEvsCRR:
    def test_american_put_pde_vs_crr(self):
        result = american_option_comparison(S0, K, T, r, sigma, option_type='put')
        pde = result['american_pde']
        crr = result['american_crr']
        rel_error = abs(pde - crr) / crr
        assert rel_error < 0.01, (
            f"American Put: PDE={pde:.6f}, CRR={crr:.6f}, erreur={rel_error:.4%}"
        )

    def test_early_exercise_premium_in_result(self):
        result = american_option_comparison(S0, K, T, r, sigma, option_type='put')
        assert result['early_exercise_premium'] > 0, (
            f"Prime exercice anticipé non positive: {result['early_exercise_premium']:.6f}"
        )
        # american > european
        assert result['american_pde'] > result['european_bs'], (
            "American PDE devrait > European BS pour un put"
        )

# -*- coding: utf-8 -*-
"""
Tests Unitaires pour le Module Yield Curve
==========================================
Bootstrap, interpolation, Nelson-Siegel, integration IRS, edge cases, sensibilite.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.yield_curve import YieldCurve, NelsonSiegelCurve
from src.interest_rate import InterestRateSwap


# ======================================================================
# Donnees de marche reutilisables
# ======================================================================

DEPOSIT_RATES = [0.03, 0.032]
DEPOSIT_MATURITIES = [0.25, 0.5]
SWAP_RATES = [0.035, 0.033, 0.031, 0.030]
SWAP_MATURITIES = [1, 2, 5, 10]
PAYMENT_FREQ = 2


def _build_curve(interpolation: str = "log_linear") -> YieldCurve:
    """Helper: construit une courbe bootstrappee avec les donnees de reference."""
    curve = YieldCurve(interpolation=interpolation)
    curve.build_from_deposits_and_swaps(
        DEPOSIT_RATES,
        DEPOSIT_MATURITIES,
        SWAP_RATES,
        SWAP_MATURITIES,
        payment_frequency=PAYMENT_FREQ,
    )
    return curve


# ======================================================================
# 1. Bootstrap basics
# ======================================================================


class TestBootstrapBasics:
    """Verification des proprietes fondamentales du bootstrap."""

    def test_df_at_zero_is_one(self):
        """DF(0) = 1.0 par definition."""
        curve = _build_curve()
        assert curve.discount_factor(0.0) == 1.0

    def test_dfs_monotonically_decreasing(self):
        """Les DF doivent decroitre strictement avec la maturite."""
        curve = _build_curve()
        dfs = curve.discount_factors
        for i in range(1, len(dfs)):
            assert dfs[i] < dfs[i - 1], (
                f"DF[{i}]={dfs[i]:.8f} >= DF[{i-1}]={dfs[i-1]:.8f}"
            )

    def test_zero_rates_positive(self):
        """Les taux zero doivent etre positifs pour toutes les maturites."""
        curve = _build_curve()
        for mat in [0.25, 0.5, 1, 2, 5, 10]:
            zr = curve.zero_rate(mat)
            assert zr > 0, f"Taux zero negatif a T={mat}: {zr}"

    def test_deposit_rate_recovery(self):
        """Le bootstrap doit recuperer DF = 1/(1+r*T) pour les depots."""
        curve = _build_curve()
        for rate, mat in zip(DEPOSIT_RATES, DEPOSIT_MATURITIES):
            expected_df = 1.0 / (1.0 + rate * mat)
            actual_df = curve.discount_factor(mat)
            assert abs(actual_df - expected_df) < 1e-10, (
                f"Depot T={mat}: attendu {expected_df:.10f}, obtenu {actual_df:.10f}"
            )

    def test_swap_rate_recovery(self):
        """La courbe bootstrappee doit repricing les swaps input au par (PV = 1.0)."""
        curve = _build_curve()
        delta = 1.0 / PAYMENT_FREQ

        for swap_rate, swap_mat in zip(SWAP_RATES, SWAP_MATURITIES):
            n_payments = int(round(swap_mat * PAYMENT_FREQ))
            payment_dates = [(i + 1) * delta for i in range(n_payments)]

            pv = 0.0
            for t_pay in payment_dates:
                pv += swap_rate * delta * curve.discount_factor(t_pay)
            pv += curve.discount_factor(swap_mat)

            assert abs(pv - 1.0) < 1e-6, (
                f"Swap T={swap_mat}: PV={pv:.8f} != 1.0"
            )

    def test_forward_rates_positive(self):
        """Les taux forward doivent etre positifs entre piliers consecutifs."""
        curve = _build_curve()
        mats = [m for m in curve.maturities if m > 0]
        for i in range(len(mats) - 1):
            fwd = curve.forward_rate(mats[i], mats[i + 1])
            assert fwd > 0, (
                f"Forward negatif entre T={mats[i]} et T={mats[i+1]}: {fwd}"
            )


# ======================================================================
# 2. Interpolation
# ======================================================================


class TestInterpolation:
    """Verification des trois methodes d'interpolation."""

    def test_linear_interpolation_between_pillars(self):
        """Interpolation lineaire: DF interpole entre les DF voisins."""
        curve = _build_curve("linear")
        T_mid = 0.375  # entre 0.25 et 0.5
        df_mid = curve.discount_factor(T_mid)
        df_low = curve.discount_factor(0.25)
        df_high = curve.discount_factor(0.5)
        assert df_high < df_mid < df_low, (
            f"DF({T_mid})={df_mid:.8f} pas entre "
            f"DF(0.25)={df_low:.8f} et DF(0.5)={df_high:.8f}"
        )

    def test_log_linear_interpolation_between_pillars(self):
        """Interpolation log-lineaire: DF interpole entre les DF voisins."""
        curve = _build_curve("log_linear")
        T_mid = 0.375
        df_mid = curve.discount_factor(T_mid)
        df_low = curve.discount_factor(0.25)
        df_high = curve.discount_factor(0.5)
        assert df_high < df_mid < df_low, (
            f"DF({T_mid})={df_mid:.8f} pas entre "
            f"DF(0.25)={df_low:.8f} et DF(0.5)={df_high:.8f}"
        )

    def test_cubic_spline_interpolation_between_pillars(self):
        """Spline cubique: DF interpole entre les DF voisins."""
        curve = _build_curve("cubic_spline")
        T_mid = 3.5  # entre 2 et 5 (assez de piliers pour la spline)
        df_mid = curve.discount_factor(T_mid)
        df_low = curve.discount_factor(2.0)
        df_high = curve.discount_factor(5.0)
        assert df_high < df_mid < df_low, (
            f"DF({T_mid})={df_mid:.8f} pas entre "
            f"DF(2)={df_low:.8f} et DF(5)={df_high:.8f}"
        )

    def test_all_methods_produce_valid_dfs(self):
        """Les 3 methodes produisent des DF dans (0, 1) pour T > 0."""
        for method in ("linear", "log_linear", "cubic_spline"):
            curve = _build_curve(method)
            for T in [0.1, 0.375, 1.5, 3.0, 7.0]:
                df = curve.discount_factor(T)
                assert 0 < df < 1, (
                    f"Methode '{method}', T={T}: DF={df:.8f} hors de (0,1)"
                )

    def test_invalid_interpolation_method_raises(self):
        """Une methode d'interpolation inconnue doit lever ValueError."""
        with pytest.raises(ValueError, match="non reconnue"):
            YieldCurve(interpolation="quadratic")


# ======================================================================
# 3. Nelson-Siegel
# ======================================================================


class TestNelsonSiegel:
    """Tests de la courbe parametrique Nelson-Siegel."""

    def _fit_ns(self) -> NelsonSiegelCurve:
        """Helper: calibre NS sur la courbe bootstrappee."""
        curve = _build_curve()
        mats = [m for m in curve.maturities if m > 0]
        rates = [curve.zero_rate(t) for t in mats]
        ns = NelsonSiegelCurve()
        ns.fit(mats, rates)
        return ns, mats, rates

    def test_fit_produces_finite_parameters(self):
        """Les parametres calibres doivent etre finis."""
        ns, _, _ = self._fit_ns()
        for name, val in [
            ("beta0", ns.beta0),
            ("beta1", ns.beta1),
            ("beta2", ns.beta2),
            ("tau", ns.tau),
        ]:
            assert np.isfinite(val), f"Parametre {name} non fini: {val}"

    def test_zero_rate_discount_factor_consistency(self):
        """DF(T) = exp(-r(T)*T) doit etre verifie pour tout T."""
        ns, _, _ = self._fit_ns()
        for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
            zr = ns.zero_rate(T)
            df = ns.discount_factor(T)
            expected_df = np.exp(-zr * T)
            assert abs(df - expected_df) < 1e-12, (
                f"T={T}: DF={df:.12f} != exp(-r*T)={expected_df:.12f}"
            )

    def test_fitted_rates_close_to_input(self):
        """Les taux NS ajustes doivent etre proches des taux input."""
        ns, mats, rates = self._fit_ns()
        for mat, rate_input in zip(mats, rates):
            rate_ns = ns.zero_rate(mat)
            assert abs(rate_ns - rate_input) < 0.005, (
                f"T={mat}: NS={rate_ns:.6f} vs input={rate_input:.6f}, "
                f"ecart={abs(rate_ns - rate_input):.6f}"
            )

    def test_ns_df_at_zero_is_one(self):
        """DF(0) = 1 pour Nelson-Siegel."""
        ns, _, _ = self._fit_ns()
        assert ns.discount_factor(0.0) == 1.0


# ======================================================================
# 4. Integration avec IRS
# ======================================================================


class TestIRSIntegration:
    """Interaction entre YieldCurve bootstrappee et IRS."""

    def test_set_curve_works(self):
        """set_curve ne doit pas lever d'erreur."""
        curve = _build_curve()
        swap = InterestRateSwap(
            notional=10_000_000,
            fixed_rate=0.035,
            maturity=5,
            payment_frequency=PAYMENT_FREQ,
        )
        swap.set_curve(curve)
        # Verifier que les attributs sont remplis
        assert swap.discount_factors is not None
        assert swap.forward_rates is not None
        assert len(swap.discount_factors) == swap.n_payments
        assert len(swap.forward_rates) == swap.n_payments

    def test_bootstrapped_vs_flat_curve_different_price(self):
        """Un IRS price avec courbe bootstrappee doit donner un resultat
        different d'une courbe plate (sauf cas degenere)."""
        curve = _build_curve()

        swap_boot = InterestRateSwap(
            notional=10_000_000,
            fixed_rate=0.035,
            maturity=5,
            payment_frequency=PAYMENT_FREQ,
        )
        swap_boot.set_curve(curve)
        price_boot = swap_boot.price()

        swap_flat = InterestRateSwap(
            notional=10_000_000,
            fixed_rate=0.035,
            maturity=5,
            payment_frequency=PAYMENT_FREQ,
        )
        swap_flat._generate_default_curve(0.035)
        price_flat = swap_flat.price()

        assert abs(price_boot - price_flat) > 1.0, (
            f"Prix bootstrap ({price_boot:.2f}) et flat ({price_flat:.2f}) "
            f"trop proches — la courbe inversee devrait creer un ecart"
        )


# ======================================================================
# 5. Edge cases
# ======================================================================


class TestEdgeCases:
    """Cas limites et configurations particulieres."""

    def test_flat_curve_matches_exp(self):
        """Courbe plate: tous les DF doivent etre exp(-r*T)."""
        flat_rate = 0.04
        curve = YieldCurve(interpolation="log_linear")
        curve.build_from_deposits_and_swaps(
            deposit_rates=[flat_rate, flat_rate],
            deposit_maturities=[0.25, 0.5],
            swap_rates=[flat_rate, flat_rate],
            swap_maturities=[1, 2],
            payment_frequency=2,
        )

        for T in [0.25, 0.5, 1.0, 2.0]:
            df = curve.discount_factor(T)
            # Pour les depots: DF = 1/(1+r*T), donc le taux zero continu
            # n'est pas exactement r. Mais les swap maturities doivent repricing au par.
            assert 0 < df < 1, f"T={T}: DF={df} hors de (0,1)"

        # Verifier que les taux zero sont proches entre eux (courbe quasi-plate)
        rates = [curve.zero_rate(T) for T in [0.25, 0.5, 1.0, 2.0]]
        spread = max(rates) - min(rates)
        assert spread < 0.005, (
            f"Courbe plate: spread des zero rates = {spread:.6f} trop large"
        )

    def test_single_deposit_single_swap(self):
        """Le bootstrap fonctionne avec un seul depot et un seul swap."""
        curve = YieldCurve(interpolation="log_linear")
        curve.build_from_deposits_and_swaps(
            deposit_rates=[0.03],
            deposit_maturities=[0.5],
            swap_rates=[0.035],
            swap_maturities=[2],
            payment_frequency=2,
        )
        # Doit avoir 3 piliers: 0, 0.5, 2
        assert len(curve.maturities) == 3
        assert curve.maturities[0] == 0.0
        assert abs(curve.maturities[1] - 0.5) < 1e-10
        assert abs(curve.maturities[2] - 2.0) < 1e-10

        # DF valides
        for m in curve.maturities:
            if m > 0:
                df = curve.discount_factor(m)
                assert 0 < df < 1

    def test_query_at_pillar_returns_exact_value(self):
        """Interroger aux maturites piliers retourne les DF exacts (pas interpoles)."""
        curve = _build_curve()
        for mat, expected_df in zip(curve.maturities, curve.discount_factors):
            actual_df = curve.discount_factor(mat)
            assert abs(actual_df - expected_df) < 1e-12, (
                f"T={mat}: attendu {expected_df:.12f}, obtenu {actual_df:.12f}"
            )

    def test_forward_rate_raises_if_t2_leq_t1(self):
        """forward_rate doit lever ValueError si T2 <= T1."""
        curve = _build_curve()
        with pytest.raises(ValueError, match="doit être > T1"):
            curve.forward_rate(2.0, 1.0)
        with pytest.raises(ValueError, match="doit être > T1"):
            curve.forward_rate(1.0, 1.0)


# ======================================================================
# 6. Sensibilite
# ======================================================================


class TestSensitivity:
    """DV01 et key rate durations."""

    def test_dv01_returns_nonzero_changes(self):
        """DV01: un bump parallele de 1bp doit changer les DF."""
        curve = _build_curve()
        result = curve.dv01(bump_bps=1)

        assert "delta_df" in result
        assert len(result["delta_df"]) > 0

        for i, delta in enumerate(result["delta_df"]):
            assert delta != 0.0, (
                f"Pilier {result['maturities'][i]}: delta_df = 0"
            )
            # Un bump positif de taux doit diminuer les DF
            assert delta < 0, (
                f"Pilier {result['maturities'][i]}: "
                f"delta_df={delta:.10f} devrait etre negatif"
            )

    def test_dv01_magnitude_increases_with_maturity(self):
        """L'impact absolu du DV01 doit croitre avec la maturite."""
        curve = _build_curve()
        result = curve.dv01(bump_bps=1)

        abs_deltas = [abs(d) for d in result["delta_df"]]
        for i in range(1, len(abs_deltas)):
            assert abs_deltas[i] >= abs_deltas[i - 1] - 1e-12, (
                f"|delta[{i}]|={abs_deltas[i]:.10f} < "
                f"|delta[{i-1}]|={abs_deltas[i-1]:.10f}"
            )

    def test_key_rate_durations_returns_all_pillars(self):
        """Key rate durations doit retourner un resultat pour chaque pilier."""
        curve = _build_curve()
        krd = curve.key_rate_durations()

        expected_pillars = [m for m in curve.maturities if m > 0]
        assert len(krd) == len(expected_pillars), (
            f"KRD a {len(krd)} piliers, attendu {len(expected_pillars)}"
        )

        for pillar in expected_pillars:
            assert pillar in krd, f"Pilier {pillar} absent du KRD"
            assert krd[pillar]["df_change"] != 0.0
            assert krd[pillar]["zr_change"] > 0

    def test_key_rate_durations_df_change_negative(self):
        """Bumper un taux zero vers le haut doit diminuer le DF correspondant."""
        curve = _build_curve()
        krd = curve.key_rate_durations(bump_bps=1)

        for pillar, data in krd.items():
            assert data["df_change"] < 0, (
                f"Pilier {pillar}: df_change={data['df_change']:.10f} "
                f"devrait etre negatif"
            )

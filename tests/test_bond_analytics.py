# -*- coding: utf-8 -*-
"""
Tests Unitaires pour Bond Analytics
====================================
Tests couvrant: pricing obligataire, duration, convexité, DV01,
analyse de chocs de taux, et portefeuille obligataire.
"""

import sys
import os
import numpy as np
import pytest

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bond_analytics import Bond, BondPortfolio, rate_shock_analysis


# ============================================================
# 1. Bond Pricing
# ============================================================

class TestBondPricing:
    """Tests pour le pricing obligataire."""

    def test_par_bond_price_equals_face_value(self):
        """
        Quand coupon_rate == ytm, le prix doit être égal à face_value.
        C'est la définition d'une obligation au pair.
        """
        bond = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        price = bond.price(ytm=0.05)
        assert abs(price - 100.0) < 1e-10, f"Prix au pair attendu 100, obtenu {price}"

    def test_par_bond_annual_coupon(self):
        """Par bond avec coupon annuel."""
        bond = Bond(face_value=1000, coupon_rate=0.08, maturity=5, frequency=1)
        price = bond.price(ytm=0.08)
        assert abs(price - 1000.0) < 1e-8

    def test_zero_coupon_bond(self):
        """
        Obligation zéro-coupon: P = FV / (1 + y/f)^n.
        Avec coupon_rate=0, FV=100, maturity=5, frequency=1, ytm=0.06:
            P = 100 / (1.06)^5 = 74.7258...
        """
        bond = Bond(face_value=100, coupon_rate=0.0, maturity=5, frequency=1)
        expected = 100 / (1.06 ** 5)
        price = bond.price(ytm=0.06)
        assert abs(price - expected) < 1e-10

    def test_price_decreases_as_yield_increases(self):
        """
        Relation inverse prix/yield: quand le yield monte, le prix baisse.
        C'est une propriété fondamentale des obligations à taux fixe.
        """
        bond = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        p_low = bond.price(ytm=0.03)
        p_mid = bond.price(ytm=0.05)
        p_high = bond.price(ytm=0.07)
        assert p_low > p_mid > p_high, (
            f"Prix doit décroître avec le yield: {p_low:.4f} > {p_mid:.4f} > {p_high:.4f}"
        )

    def test_premium_bond(self):
        """Quand coupon > yield, obligation en prime (prix > face_value)."""
        bond = Bond(face_value=100, coupon_rate=0.08, maturity=10, frequency=2)
        price = bond.price(ytm=0.05)
        assert price > 100.0, f"Obligation en prime attendue, obtenu {price:.4f}"

    def test_discount_bond(self):
        """Quand coupon < yield, obligation en décote (prix < face_value)."""
        bond = Bond(face_value=100, coupon_rate=0.03, maturity=10, frequency=2)
        price = bond.price(ytm=0.05)
        assert price < 100.0, f"Obligation en décote attendue, obtenu {price:.4f}"

    def test_different_face_values(self):
        """Le prix scale linéairement avec face_value."""
        bond_100 = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        bond_1000 = Bond(face_value=1000, coupon_rate=0.05, maturity=10, frequency=2)
        assert abs(bond_1000.price(ytm=0.06) - 10 * bond_100.price(ytm=0.06)) < 1e-8

    def test_quarterly_coupon_par(self):
        """Par bond avec coupon trimestriel."""
        bond = Bond(face_value=100, coupon_rate=0.04, maturity=3, frequency=4)
        price = bond.price(ytm=0.04)
        assert abs(price - 100.0) < 1e-8


# ============================================================
# 2. Duration
# ============================================================

class TestDuration:
    """Tests pour Macaulay Duration et Modified Duration."""

    def test_macaulay_less_than_maturity(self):
        """
        Macaulay Duration d'une obligation avec coupon est toujours
        strictement inférieure à sa maturité (les coupons intermédiaires
        rapprochent la durée moyenne des flux).
        """
        bond = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        d_mac = bond.macaulay_duration(ytm=0.05)
        assert 0 < d_mac < bond.maturity, (
            f"Macaulay Duration ({d_mac:.4f}) doit être dans (0, {bond.maturity})"
        )

    def test_zero_coupon_macaulay_equals_maturity(self):
        """
        Pour un zéro-coupon, Macaulay Duration = maturité exactement.
        Le seul flux est le remboursement à maturité.
        """
        bond = Bond(face_value=100, coupon_rate=0.0, maturity=7, frequency=1)
        d_mac = bond.macaulay_duration(ytm=0.05)
        assert abs(d_mac - 7.0) < 1e-10, (
            f"Zéro-coupon: Macaulay Duration attendue 7.0, obtenu {d_mac:.4f}"
        )

    def test_zero_coupon_macaulay_semiannual(self):
        """Zéro-coupon semestriel: Macaulay Duration = maturité."""
        bond = Bond(face_value=100, coupon_rate=0.0, maturity=5, frequency=2)
        d_mac = bond.macaulay_duration(ytm=0.04)
        assert abs(d_mac - 5.0) < 1e-10

    def test_modified_equals_macaulay_divided(self):
        """
        Relation fondamentale: D_mod = D_mac / (1 + y/f).
        Vérifie la cohérence entre les deux implémentations.
        """
        bond = Bond(face_value=100, coupon_rate=0.06, maturity=10, frequency=2)
        ytm = 0.05
        d_mac = bond.macaulay_duration(ytm)
        d_mod = bond.modified_duration(ytm)
        expected_d_mod = d_mac / (1 + ytm / bond.frequency)
        assert abs(d_mod - expected_d_mod) < 1e-12, (
            f"D_mod ({d_mod}) != D_mac/(1+y/f) ({expected_d_mod})"
        )

    def test_higher_coupon_lower_duration(self):
        """
        Un coupon plus élevé raccourcit la Macaulay Duration
        (plus de flux intermédiaires pondèrent les périodes proches).
        """
        bond_low = Bond(face_value=100, coupon_rate=0.02, maturity=10, frequency=2)
        bond_high = Bond(face_value=100, coupon_rate=0.08, maturity=10, frequency=2)
        ytm = 0.05
        assert bond_low.macaulay_duration(ytm) > bond_high.macaulay_duration(ytm)

    def test_longer_maturity_higher_duration(self):
        """Maturité plus longue implique duration plus élevée (même coupon/yield)."""
        bond_5y = Bond(face_value=100, coupon_rate=0.05, maturity=5, frequency=2)
        bond_30y = Bond(face_value=100, coupon_rate=0.05, maturity=30, frequency=2)
        ytm = 0.05
        assert bond_5y.macaulay_duration(ytm) < bond_30y.macaulay_duration(ytm)

    def test_modified_duration_positive(self):
        """Modified Duration est toujours positive."""
        bond = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        assert bond.modified_duration(ytm=0.05) > 0


# ============================================================
# 3. Convexity
# ============================================================

class TestConvexity:
    """Tests pour la convexité."""

    def test_convexity_always_positive(self):
        """
        La convexité est toujours positive pour une obligation standard
        (tous les flux sont positifs → dérivée seconde du prix > 0).
        """
        bond = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        conv = bond.convexity(ytm=0.05)
        assert conv > 0, f"Convexité doit être positive, obtenu {conv}"

    def test_convexity_positive_zero_coupon(self):
        """Convexité positive même pour un zéro-coupon."""
        bond = Bond(face_value=100, coupon_rate=0.0, maturity=10, frequency=1)
        conv = bond.convexity(ytm=0.05)
        assert conv > 0

    def test_longer_maturity_higher_convexity(self):
        """
        Maturité plus longue → convexité plus élevée.
        Les flux éloignés contribuent davantage à la courbure.
        """
        bond_5y = Bond(face_value=100, coupon_rate=0.05, maturity=5, frequency=2)
        bond_20y = Bond(face_value=100, coupon_rate=0.05, maturity=20, frequency=2)
        ytm = 0.05
        assert bond_5y.convexity(ytm) < bond_20y.convexity(ytm)

    def test_lower_coupon_higher_convexity(self):
        """Coupon plus bas → convexité plus élevée (même maturité/yield)."""
        bond_low = Bond(face_value=100, coupon_rate=0.02, maturity=10, frequency=2)
        bond_high = Bond(face_value=100, coupon_rate=0.08, maturity=10, frequency=2)
        ytm = 0.05
        assert bond_low.convexity(ytm) > bond_high.convexity(ytm)

    def test_convexity_numerical_accuracy(self):
        """
        Vérifie que la convexité numérique est cohérente avec la formule
        analytique pour un zéro-coupon:
            C_zc = n*(n+1) / [f^2 * (1+y/f)^2]
        où n = nombre de périodes.
        """
        bond = Bond(face_value=100, coupon_rate=0.0, maturity=5, frequency=1)
        ytm = 0.06
        n = 5
        f = 1
        expected = n * (n + 1) / (f ** 2 * (1 + ytm / f) ** 2)
        conv = bond.convexity(ytm)
        # Tolérance plus large pour méthode numérique
        assert abs(conv - expected) / expected < 0.001, (
            f"Convexité numérique ({conv:.4f}) vs analytique ({expected:.4f})"
        )


# ============================================================
# 4. DV01
# ============================================================

class TestDV01:
    """Tests pour DV01 (Dollar Value of One Basis Point)."""

    def test_dv01_positive(self):
        """DV01 est toujours positif (prix baisse quand yield monte)."""
        bond = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        dv01 = bond.dv01(ytm=0.05)
        assert dv01 > 0, f"DV01 doit être positif, obtenu {dv01}"

    def test_dv01_formula(self):
        """
        Vérifie DV01 = D_mod * P * 0.0001.
        """
        bond = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        ytm = 0.05
        expected = bond.modified_duration(ytm) * bond.price(ytm) * 0.0001
        dv01 = bond.dv01(ytm)
        assert abs(dv01 - expected) < 1e-12

    def test_dv01_approximates_actual_price_change(self):
        """
        DV01 doit bien approximer |P(y+1bp) - P(y)|.
        L'erreur vient de la convexité (2ème ordre), négligeable à 1bp.
        """
        bond = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        ytm = 0.05
        dv01 = bond.dv01(ytm)
        actual_change = abs(bond.price(ytm + 0.0001) - bond.price(ytm))
        # À 1bp l'approximation linéaire est très précise
        assert abs(dv01 - actual_change) / actual_change < 0.001

    def test_dv01_longer_maturity_higher(self):
        """DV01 plus élevé pour maturité plus longue (plus de risque de taux)."""
        bond_5y = Bond(face_value=100, coupon_rate=0.05, maturity=5, frequency=2)
        bond_20y = Bond(face_value=100, coupon_rate=0.05, maturity=20, frequency=2)
        ytm = 0.05
        assert bond_5y.dv01(ytm) < bond_20y.dv01(ytm)

    def test_dv01_scales_with_face_value(self):
        """DV01 scale linéairement avec face_value."""
        bond_100 = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        bond_1000 = Bond(face_value=1000, coupon_rate=0.05, maturity=10, frequency=2)
        ytm = 0.05
        assert abs(bond_1000.dv01(ytm) - 10 * bond_100.dv01(ytm)) < 1e-10


# ============================================================
# 5. Rate Shock Analysis
# ============================================================

class TestRateShockAnalysis:
    """Tests pour l'analyse de chocs de taux."""

    def setup_method(self):
        """Bond standard pour les tests de chocs."""
        self.bond = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        self.ytm = 0.05

    def test_returns_dataframe_with_correct_columns(self):
        """Vérifie la structure du DataFrame retourné."""
        df = rate_shock_analysis(self.bond, self.ytm)
        expected_cols = {
            "shock_bps", "actual_pnl", "duration_approx",
            "dur_conv_approx", "error_duration", "error_dur_conv",
        }
        assert set(df.columns) == expected_cols

    def test_default_shocks(self):
        """6 chocs par défaut: ±50, ±100, ±200 bp."""
        df = rate_shock_analysis(self.bond, self.ytm)
        assert len(df) == 6
        assert sorted(df["shock_bps"].tolist()) == [-200, -100, -50, 50, 100, 200]

    def test_custom_shocks(self):
        """Chocs personnalisés."""
        shocks = [-0.03, -0.01, 0.01, 0.03]
        df = rate_shock_analysis(self.bond, self.ytm, shocks=shocks)
        assert len(df) == 4

    def test_duration_error_grows_quadratically(self):
        """
        L'erreur de l'approximation Duration (1er ordre) croît
        quadratiquement avec la taille du choc — c'est le terme de
        convexité manquant (1/2 * C * dy^2).
        """
        shocks = [0.005, 0.01, 0.02]  # 50bp, 100bp, 200bp
        df = rate_shock_analysis(self.bond, self.ytm, shocks=shocks)

        errors = df["error_duration"].abs().tolist()
        # Ratio d'erreur entre chocs doublés devrait être ~4x (quadratique)
        ratio_1 = errors[1] / errors[0]  # 100bp/50bp → ~4x
        ratio_2 = errors[2] / errors[1]  # 200bp/100bp → ~4x
        assert 3.0 < ratio_1 < 5.0, f"Ratio 100bp/50bp = {ratio_1:.2f}, attendu ~4"
        assert 3.0 < ratio_2 < 5.0, f"Ratio 200bp/100bp = {ratio_2:.2f}, attendu ~4"

    def test_dur_conv_much_better_for_large_shocks(self):
        """
        Pour des chocs de ±200bp, l'approximation Duration+Convexité
        est beaucoup plus précise que Duration seule.
        """
        shocks = [-0.02, 0.02]  # ±200bp
        df = rate_shock_analysis(self.bond, self.ytm, shocks=shocks)

        for _, row in df.iterrows():
            err_dur = abs(row["error_duration"])
            err_dur_conv = abs(row["error_dur_conv"])
            assert err_dur_conv < err_dur * 0.1, (
                f"À {row['shock_bps']}bp: erreur dur+conv ({err_dur_conv:.6f}) "
                f"devrait être << erreur dur ({err_dur:.6f})"
            )

    def test_actual_pnl_sign(self):
        """
        Yield en hausse → prix baisse → PnL négatif.
        Yield en baisse → prix monte → PnL positif.
        """
        shocks = [-0.01, 0.01]
        df = rate_shock_analysis(self.bond, self.ytm, shocks=shocks)
        neg_shock = df[df["shock_bps"] == -100].iloc[0]
        pos_shock = df[df["shock_bps"] == 100].iloc[0]
        assert neg_shock["actual_pnl"] > 0, "Yield baisse → PnL positif"
        assert pos_shock["actual_pnl"] < 0, "Yield hausse → PnL négatif"

    def test_error_consistency(self):
        """Vérifie que error = actual_pnl - approx."""
        df = rate_shock_analysis(self.bond, self.ytm)
        for _, row in df.iterrows():
            assert abs(
                row["error_duration"] - (row["actual_pnl"] - row["duration_approx"])
            ) < 1e-12
            assert abs(
                row["error_dur_conv"] - (row["actual_pnl"] - row["dur_conv_approx"])
            ) < 1e-12


# ============================================================
# 6. Portfolio
# ============================================================

class TestBondPortfolio:
    """Tests pour BondPortfolio."""

    def test_single_bond_portfolio(self):
        """Portfolio avec une seule obligation = obligation elle-même."""
        bond = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        ytm = 0.05

        portfolio = BondPortfolio()
        portfolio.add_bond(bond, quantity=1, ytm=ytm)

        assert abs(portfolio.total_value() - bond.price(ytm)) < 1e-10
        assert abs(portfolio.portfolio_duration() - bond.modified_duration(ytm)) < 1e-10
        assert abs(portfolio.portfolio_convexity() - bond.convexity(ytm)) < 1e-10

    def test_portfolio_total_value(self):
        """Valeur totale = somme des (prix × quantité)."""
        bond1 = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        bond2 = Bond(face_value=100, coupon_rate=0.03, maturity=5, frequency=2)

        portfolio = BondPortfolio()
        portfolio.add_bond(bond1, quantity=100, ytm=0.05)
        portfolio.add_bond(bond2, quantity=200, ytm=0.03)

        expected = 100 * bond1.price(0.05) + 200 * bond2.price(0.03)
        assert abs(portfolio.total_value() - expected) < 1e-8

    def test_portfolio_weighted_average_duration(self):
        """
        Duration du portefeuille = moyenne pondérée par la valeur
        des modified durations individuelles.
        """
        bond1 = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        bond2 = Bond(face_value=100, coupon_rate=0.03, maturity=5, frequency=2)
        ytm1, ytm2 = 0.05, 0.03

        portfolio = BondPortfolio()
        portfolio.add_bond(bond1, quantity=50, ytm=ytm1)
        portfolio.add_bond(bond2, quantity=50, ytm=ytm2)

        v1 = 50 * bond1.price(ytm1)
        v2 = 50 * bond2.price(ytm2)
        total = v1 + v2
        expected_dur = (
            (v1 / total) * bond1.modified_duration(ytm1)
            + (v2 / total) * bond2.modified_duration(ytm2)
        )
        assert abs(portfolio.portfolio_duration() - expected_dur) < 1e-10

    def test_portfolio_duration_between_individual(self):
        """
        La duration du portefeuille est bornée entre la plus petite
        et la plus grande des durations individuelles.
        """
        bond_short = Bond(face_value=100, coupon_rate=0.05, maturity=2, frequency=2)
        bond_long = Bond(face_value=100, coupon_rate=0.05, maturity=20, frequency=2)
        ytm = 0.05

        portfolio = BondPortfolio()
        portfolio.add_bond(bond_short, quantity=100, ytm=ytm)
        portfolio.add_bond(bond_long, quantity=100, ytm=ytm)

        d_short = bond_short.modified_duration(ytm)
        d_long = bond_long.modified_duration(ytm)
        d_port = portfolio.portfolio_duration()

        assert d_short < d_port < d_long

    def test_empty_portfolio(self):
        """Portefeuille vide: valeur=0, duration=0, convexité=0."""
        portfolio = BondPortfolio()
        assert portfolio.total_value() == 0.0
        assert portfolio.portfolio_duration() == 0.0
        assert portfolio.portfolio_convexity() == 0.0

    def test_rate_shock_analysis_portfolio(self):
        """rate_shock_analysis fonctionne aussi avec un portefeuille."""
        bond1 = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        bond2 = Bond(face_value=100, coupon_rate=0.03, maturity=5, frequency=2)

        portfolio = BondPortfolio()
        portfolio.add_bond(bond1, quantity=100, ytm=0.05)
        portfolio.add_bond(bond2, quantity=200, ytm=0.05)

        df = rate_shock_analysis(portfolio, ytm=0.05)
        assert len(df) == 6
        # Vérifier que le PnL a le bon signe
        pos_shock = df[df["shock_bps"] > 0]
        neg_shock = df[df["shock_bps"] < 0]
        assert all(pos_shock["actual_pnl"] < 0)
        assert all(neg_shock["actual_pnl"] > 0)


# ============================================================
# Validation des entrées
# ============================================================

class TestInputValidation:
    """Tests pour la validation des paramètres."""

    def test_negative_face_value(self):
        with pytest.raises(ValueError, match="face_value"):
            Bond(face_value=-100, coupon_rate=0.05, maturity=10, frequency=2)

    def test_negative_coupon_rate(self):
        with pytest.raises(ValueError, match="coupon_rate"):
            Bond(face_value=100, coupon_rate=-0.01, maturity=10, frequency=2)

    def test_zero_maturity(self):
        with pytest.raises(ValueError, match="maturity"):
            Bond(face_value=100, coupon_rate=0.05, maturity=0, frequency=2)

    def test_invalid_frequency(self):
        with pytest.raises(ValueError, match="frequency"):
            Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=3)

    def test_description(self):
        bond = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
        desc = bond.description()
        assert "100" in desc
        assert "5.00%" in desc
        assert "semestriel" in desc
        assert "10" in desc

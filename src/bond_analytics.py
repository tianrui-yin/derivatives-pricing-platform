# -*- coding: utf-8 -*-
"""
Bond Analytics
==============
Pricing obligataire, mesures de risque et analyse de chocs de taux.

Produits implementes:
    - Bond: Obligation a taux fixe (pricing, duration, convexite, DV01)
    - BondPortfolio: Portefeuille obligataire (risque agrege)
    - rate_shock_analysis: Comparaison des approximations Taylor

Reference: Hull, Chapitres 4 (Duration, Convexite) et 6 (Pricing obligataire)

Analogie cle:
    Duration/Convexity pour les obligations ~ Delta/Gamma pour les options.
    L'approximation de 1er ordre (Duration) est lineaire, celle de 2eme ordre
    (Duration + Convexity) corrige la courbure, tout comme Delta + Gamma.
"""

import numpy as np
import pandas as pd


class Bond:
    """
    Obligation a taux fixe: pricing et analytics.

    Parametres:
    -----------
    face_value : float
        Valeur nominale / pair (defaut 100)
    coupon_rate : float
        Taux de coupon annuel (ex: 0.05 = 5%)
    maturity : float
        Maturite en annees
    frequency : int
        Nombre de coupons par an (1=annuel, 2=semestriel)

    Exemple:
    --------
    >>> bond = Bond(face_value=100, coupon_rate=0.05, maturity=10, frequency=2)
    >>> print(f"Prix au pair: {bond.price(ytm=0.05):.2f}")
    Prix au pair: 100.00
    >>> print(f"Duration modifiee: {bond.modified_duration(ytm=0.05):.4f}")
    """

    def __init__(self, face_value=100, coupon_rate=0.05, maturity=10, frequency=2):
        if face_value <= 0:
            raise ValueError("face_value doit etre positif")
        if coupon_rate < 0:
            raise ValueError("coupon_rate doit etre >= 0")
        if maturity <= 0:
            raise ValueError("maturity doit etre positif")
        if frequency not in (1, 2, 4, 12):
            raise ValueError("frequency doit etre 1, 2, 4 ou 12")

        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.frequency = frequency

        # Nombre total de periodes et coupon par periode
        self.n_periods = int(maturity * frequency)
        self.coupon = face_value * coupon_rate / frequency

    def price(self, ytm):
        """
        Prix de l'obligation par actualisation des flux.

        Formule (compounding discret):
            P = sum_{i=1}^{n} C / (1 + y/f)^i  +  FV / (1 + y/f)^n

        ou C = face_value * coupon_rate / frequency,
           n = maturity * frequency,
           f = frequency.

        Parametres:
        -----------
        ytm : float
            Yield to maturity (taux actuariel)

        Retourne:
        ---------
        float : prix de l'obligation
        """
        y_per_period = ytm / self.frequency
        discount = 1 + y_per_period

        pv = 0.0
        for i in range(1, self.n_periods + 1):
            pv += self.coupon / discount ** i

        # Principal a maturite
        pv += self.face_value / discount ** self.n_periods

        return pv

    def macaulay_duration(self, ytm):
        """
        Macaulay Duration: moyenne ponderee du temps des flux.

        Formule:
            D_mac = (1/P) * [ sum_{i=1}^{n} (i/f) * C / (1+y/f)^i
                              + (n/f) * FV / (1+y/f)^n ]

        Parametres:
        -----------
        ytm : float
            Yield to maturity

        Retourne:
        ---------
        float : Macaulay Duration (en annees)
        """
        y_per_period = ytm / self.frequency
        discount = 1 + y_per_period
        p = self.price(ytm)

        weighted_sum = 0.0
        for i in range(1, self.n_periods + 1):
            t_i = i / self.frequency  # temps en annees
            pv_cf = self.coupon / discount ** i
            weighted_sum += t_i * pv_cf

        # Contribution du principal
        t_n = self.n_periods / self.frequency
        pv_principal = self.face_value / discount ** self.n_periods
        weighted_sum += t_n * pv_principal

        return weighted_sum / p

    def modified_duration(self, ytm):
        """
        Modified Duration: sensibilite du prix au yield.

        Formule:
            D_mod = D_mac / (1 + y/f)

        Approximation:
            dP/P ~ -D_mod * dy

        Parametres:
        -----------
        ytm : float
            Yield to maturity

        Retourne:
        ---------
        float : Modified Duration
        """
        d_mac = self.macaulay_duration(ytm)
        return d_mac / (1 + ytm / self.frequency)

    def convexity(self, ytm):
        """
        Convexite: correction de 2eme ordre pour l'approximation de Duration.

        Methode numerique (derivee seconde):
            C = [P(y+dy) + P(y-dy) - 2*P(y)] / [P(y) * dy^2]

        avec dy = 1bp (0.0001).

        L'approximation complete:
            dP/P ~ -D_mod * dy + (1/2) * C * dy^2

        Parametres:
        -----------
        ytm : float
            Yield to maturity

        Retourne:
        ---------
        float : Convexite
        """
        dy = 0.0001  # 1bp
        p = self.price(ytm)
        p_up = self.price(ytm + dy)
        p_down = self.price(ytm - dy)

        return (p_up + p_down - 2 * p) / (p * dy ** 2)

    def dv01(self, ytm):
        """
        DV01 (Dollar Value of One Basis Point).

        Formule:
            DV01 = D_mod * P * 0.0001

        Mesure le changement absolu de prix pour 1bp de mouvement du yield.

        Parametres:
        -----------
        ytm : float
            Yield to maturity

        Retourne:
        ---------
        float : DV01 (valeur absolue)
        """
        return self.modified_duration(ytm) * self.price(ytm) * 0.0001

    def description(self):
        """Description de l'obligation."""
        freq_name = {1: "annuel", 2: "semestriel", 4: "trimestriel", 12: "mensuel"}
        return (
            f"Obligation: FV={self.face_value:,.0f}, "
            f"Coupon={self.coupon_rate:.2%} ({freq_name.get(self.frequency, '')}), "
            f"Maturite={self.maturity}ans"
        )


class BondPortfolio:
    """
    Portefeuille d'obligations pour analyse de risque agregee.

    Exemple:
    --------
    >>> portfolio = BondPortfolio()
    >>> portfolio.add_bond(Bond(100, 0.05, 10, 2), quantity=100, ytm=0.05)
    >>> portfolio.add_bond(Bond(100, 0.03, 5, 2), quantity=200, ytm=0.03)
    >>> print(f"Valeur: {portfolio.total_value():,.2f}")
    >>> print(f"Duration: {portfolio.portfolio_duration():.4f}")
    """

    def __init__(self):
        self.bonds = []  # list of (Bond, quantity, ytm)

    def add_bond(self, bond, quantity=1, ytm=0.05):
        """
        Ajoute une obligation au portefeuille.

        Parametres:
        -----------
        bond : Bond
            Obligation a ajouter
        quantity : int
            Nombre d'obligations
        ytm : float
            Yield to maturity de cette obligation
        """
        self.bonds.append((bond, quantity, ytm))

    def total_value(self):
        """
        Valeur totale du portefeuille.

        Retourne:
        ---------
        float : somme des (prix * quantite) pour chaque obligation
        """
        if not self.bonds:
            return 0.0
        return sum(
            quantity * bond.price(ytm)
            for bond, quantity, ytm in self.bonds
        )

    def portfolio_duration(self):
        """
        Modified Duration du portefeuille (moyenne ponderee par la valeur).

        Formule:
            D_portfolio = sum(w_i * D_mod_i)
            ou w_i = V_i / V_total

        Retourne:
        ---------
        float : Modified Duration du portefeuille
        """
        total = self.total_value()
        if total == 0:
            return 0.0

        weighted_dur = sum(
            quantity * bond.price(ytm) * bond.modified_duration(ytm)
            for bond, quantity, ytm in self.bonds
        )
        return weighted_dur / total

    def portfolio_convexity(self):
        """
        Convexite du portefeuille (moyenne ponderee par la valeur).

        Formule:
            C_portfolio = sum(w_i * C_i)

        Retourne:
        ---------
        float : Convexite du portefeuille
        """
        total = self.total_value()
        if total == 0:
            return 0.0

        weighted_conv = sum(
            quantity * bond.price(ytm) * bond.convexity(ytm)
            for bond, quantity, ytm in self.bonds
        )
        return weighted_conv / total


def rate_shock_analysis(bond_or_portfolio, ytm, shocks=None):
    """
    Comparaison de trois methodes d'approximation sous chocs de taux.

    Methodes:
        1. Duration (1er ordre):     dP ~ -D_mod * P * dy
        2. Duration + Convexite (2e): dP ~ -D_mod * P * dy + (1/2) * C * P * dy^2
        3. Full repricing:            P(y + dy) - P(y)

    Ceci demontre que pour des mouvements larges, la correction de 2eme ordre
    (Duration + Convexite) est beaucoup plus precise — exactement analogue
    a l'approximation Delta vs Delta + Gamma pour les options.

    Parametres:
    -----------
    bond_or_portfolio : Bond ou BondPortfolio
        Obligation ou portefeuille a analyser
    ytm : float
        Yield to maturity de base
    shocks : list of float, optional
        Chocs en fraction (ex: 0.01 = 100bp).
        Defaut: [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]

    Retourne:
    ---------
    pd.DataFrame avec colonnes:
        shock_bps, actual_pnl, duration_approx, dur_conv_approx,
        error_duration, error_dur_conv
    """
    if shocks is None:
        shocks = [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]

    # Calcul des metriques de base
    if isinstance(bond_or_portfolio, Bond):
        p = bond_or_portfolio.price(ytm)
        d_mod = bond_or_portfolio.modified_duration(ytm)
        conv = bond_or_portfolio.convexity(ytm)
        price_func = bond_or_portfolio.price
    else:
        # BondPortfolio
        p = bond_or_portfolio.total_value()
        d_mod = bond_or_portfolio.portfolio_duration()
        conv = bond_or_portfolio.portfolio_convexity()
        # Pour le repricing du portfolio, on doit recalculer avec un shift parallele
        def price_func(y):
            return sum(
                qty * b.price(y)
                for b, qty, _ in bond_or_portfolio.bonds
            )

    results = []
    for dy in shocks:
        shock_bps = int(round(dy * 10000))

        # Full repricing
        p_shocked = price_func(ytm + dy)
        actual_pnl = p_shocked - p

        # Approximation 1er ordre (Duration)
        dur_approx = -d_mod * p * dy

        # Approximation 2eme ordre (Duration + Convexite)
        dur_conv_approx = -d_mod * p * dy + 0.5 * conv * p * dy ** 2

        results.append({
            "shock_bps": shock_bps,
            "actual_pnl": actual_pnl,
            "duration_approx": dur_approx,
            "dur_conv_approx": dur_conv_approx,
            "error_duration": actual_pnl - dur_approx,
            "error_dur_conv": actual_pnl - dur_conv_approx,
        })

    return pd.DataFrame(results)

# -*- coding: utf-8 -*-
"""
Construction de Courbe des Taux (Yield Curve)
=============================================
Bootstrap à partir d'instruments de marché et courbe paramétrique Nelson-Siegel.

Référence: Hull, Chapitres 4 (Taux d'intérêt) et 7 (Swaps)

Pipeline de bootstrap:
    1. Court terme (0-1an) : taux de dépôt → facteurs d'actualisation (simple compounding)
    2. Long terme (1an+) : taux swap → résolution itérative des facteurs d'actualisation

Méthodes d'interpolation:
    - 'linear' : interpolation linéaire sur les taux zéro
    - 'log_linear' : interpolation linéaire sur log(DF)
    - 'cubic_spline' : spline cubique sur les taux zéro
"""

import logging
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

VALID_INTERPOLATIONS = ('linear', 'log_linear', 'cubic_spline')


class YieldCurve:
    """
    Courbe des taux bootstrappée à partir d'instruments de marché.

    Le bootstrap détermine les facteurs d'actualisation (discount factors) à
    chaque maturité pilier, puis interpole pour les maturités intermédiaires.

    Attributs:
    ----------
    maturities : list[float]
        Maturités piliers en années
    discount_factors : list[float]
        Facteurs d'actualisation à chaque pilier
    zero_rates : list[float]
        Taux zéro (composé continu) à chaque pilier
    interpolation : str
        Méthode d'interpolation ('linear', 'log_linear', 'cubic_spline')

    Exemple:
    --------
    >>> curve = YieldCurve(interpolation='log_linear')
    >>> curve.build_from_deposits_and_swaps(
    ...     deposit_rates=[0.03, 0.032],
    ...     deposit_maturities=[0.25, 0.5],
    ...     swap_rates=[0.035, 0.033],
    ...     swap_maturities=[1, 2],
    ...     payment_frequency=2
    ... )
    >>> print(f"DF(1Y) = {curve.discount_factor(1.0):.6f}")
    """

    def __init__(self, interpolation: str = 'log_linear'):
        """
        Initialise la courbe des taux.

        Paramètres:
        -----------
        interpolation : str
            Méthode d'interpolation parmi 'linear', 'log_linear', 'cubic_spline'

        Raises:
        -------
        ValueError
            Si la méthode d'interpolation n'est pas reconnue
        """
        if interpolation not in VALID_INTERPOLATIONS:
            raise ValueError(
                f"Interpolation '{interpolation}' non reconnue. "
                f"Choix: {VALID_INTERPOLATIONS}"
            )
        self.maturities: list[float] = []
        self.discount_factors: list[float] = []
        self.zero_rates: list[float] = []
        self.interpolation = interpolation
        self._spline: Optional[CubicSpline] = None

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def build_from_deposits_and_swaps(
        self,
        deposit_rates: list[float],
        deposit_maturities: list[float],
        swap_rates: list[float],
        swap_maturities: list[float],
        payment_frequency: int = 2,
    ) -> None:
        """
        Bootstrap de la courbe à partir de taux de dépôt et taux swap.

        Étape 1 - Dépôts (court terme):
            DF(T) = 1 / (1 + r * T)  (convention money market, simple compounding)

        Étape 2 - Swaps (long terme):
            Pour chaque maturité swap T_n, le taux swap c_n satisfait :
                sum(c_n * delta * DF(T_i), i=1..n) + DF(T_n) = 1
            D'où :
                DF(T_n) = (1 - c_n * sum(delta * DF(T_i), i=1..n-1)) / (1 + c_n * delta)

        Paramètres:
        -----------
        deposit_rates : list[float]
            Taux de dépôt annualisés (ex: [0.03, 0.032])
        deposit_maturities : list[float]
            Maturités des dépôts en années (ex: [0.25, 0.5])
        swap_rates : list[float]
            Taux swap par (ex: [0.035, 0.033, 0.031])
        swap_maturities : list[float]
            Maturités des swaps en années (ex: [1, 2, 5])
        payment_frequency : int
            Fréquence de paiement des coupons swap par an (2 = semestriel)
        """
        self.maturities = [0.0]
        self.discount_factors = [1.0]

        # --- Étape 1 : Dépôts ---
        for rate, mat in zip(deposit_rates, deposit_maturities):
            df = 1.0 / (1.0 + rate * mat)
            self.maturities.append(mat)
            self.discount_factors.append(df)
            logger.debug("Depot T=%.2f: DF=%.8f", mat, df)

        # --- Étape 2 : Swaps (bootstrap itératif) ---
        # Pour chaque swap, résoudre DF(T_n) tel que le swap se price au par.
        # Difficulté : les paiements intermédiaires (ex: T=1.5 pour un swap 2Y)
        # sont interpolés entre piliers. Quand on ajoute T_n comme nouveau pilier,
        # l'interpolation des dates entre l'ancien dernier pilier et T_n change.
        # Solution : utiliser un solveur numérique (bisection) pour assurer la
        # cohérence entre le DF bootstrappé et l'interpolation finale.
        delta = 1.0 / payment_frequency

        for swap_rate, swap_mat in zip(swap_rates, swap_maturities):
            n_payments = int(round(swap_mat * payment_frequency))
            payment_dates = [(i + 1) * delta for i in range(n_payments)]

            def swap_par_equation(df_candidate):
                """Retourne V du swap si DF(T_n) = df_candidate.

                On ajoute temporairement (swap_mat, df_candidate) aux piliers,
                puis on évalue sum(c*delta*DF(T_i)) + DF(T_n) - 1 = 0.
                """
                # Ajouter temporairement le candidat
                self.maturities.append(swap_mat)
                self.discount_factors.append(df_candidate)
                # Recalculer les zero rates temporaires pour l'interpolation
                self._rebuild_zero_rates()
                if self.interpolation == 'cubic_spline':
                    self._rebuild_spline()

                total = 0.0
                for t_pay in payment_dates:
                    df_t = self.discount_factor(t_pay)
                    total += swap_rate * delta * df_t
                total += self.discount_factor(swap_mat)

                # Retirer le candidat temporaire
                self.maturities.pop()
                self.discount_factors.pop()
                self._rebuild_zero_rates()
                if self.interpolation == 'cubic_spline':
                    self._rebuild_spline()

                return total - 1.0

            # Bisection : DF doit être dans (0, 1)
            lo, hi = 0.01, 1.0
            for _ in range(100):
                mid = (lo + hi) / 2.0
                val = swap_par_equation(mid)
                if val > 0:
                    # total trop grand → DF trop grand → réduire
                    hi = mid
                else:
                    lo = mid
                if abs(val) < 1e-14:
                    break
            df_new = (lo + hi) / 2.0

            self.maturities.append(swap_mat)
            self.discount_factors.append(df_new)
            self._rebuild_zero_rates()
            logger.debug("Swap T=%.1f (rate=%.4f): DF=%.8f", swap_mat, swap_rate, df_new)

        # Calculer les taux zéro et construire la spline
        self._rebuild_zero_rates()
        self._rebuild_spline()

        logger.info(
            "Courbe construite: %d piliers, T_max=%.1f",
            len(self.maturities), max(self.maturities)
        )

    # ------------------------------------------------------------------
    # Interpolation interne (utilisée pendant le bootstrap)
    # ------------------------------------------------------------------

    def _interpolate_df(self, T: float) -> float:
        """
        Interpole le DF à la maturité T en utilisant les piliers déjà construits.

        Utilisé pendant le bootstrap pour les dates de paiement intermédiaires.
        Utilise l'interpolation log-linéaire sur les DF (la plus stable pour le bootstrap).

        Paramètres:
        -----------
        T : float
            Maturité cible

        Retourne:
        ---------
        float : facteur d'actualisation interpolé
        """
        if T <= 0:
            return 1.0

        mats = self.maturities
        dfs = self.discount_factors

        # Vérifier si T correspond exactement à un pilier
        for i, m in enumerate(mats):
            if abs(m - T) < 1e-10:
                return dfs[i]

        # Trouver les piliers encadrants
        idx_lower = 0
        idx_upper = len(mats) - 1
        for i in range(len(mats)):
            if mats[i] <= T:
                idx_lower = i
            if mats[i] >= T and i < idx_upper:
                idx_upper = i
                break

        if idx_lower == idx_upper:
            return dfs[idx_lower]

        # Interpolation log-linéaire sur les DF
        t1, t2 = mats[idx_lower], mats[idx_upper]
        df1, df2 = dfs[idx_lower], dfs[idx_upper]

        if t2 - t1 < 1e-12:
            return df1

        weight = (T - t1) / (t2 - t1)
        log_df = np.log(df1) * (1 - weight) + np.log(df2) * weight
        return np.exp(log_df)

    # ------------------------------------------------------------------
    # Méthodes publiques d'interrogation
    # ------------------------------------------------------------------

    def discount_factor(self, T: float) -> float:
        """
        Facteur d'actualisation interpolé à la maturité T.

        Paramètres:
        -----------
        T : float
            Maturité en années

        Retourne:
        ---------
        float : DF(T)
        """
        if T <= 0:
            return 1.0

        mats = self.maturities
        dfs = self.discount_factors

        # Correspondance exacte avec un pilier
        for i, m in enumerate(mats):
            if abs(m - T) < 1e-10:
                return dfs[i]

        # Extrapolation (flat forward) au-delà du dernier pilier
        if T > max(mats):
            last_zr = self.zero_rates[-1]
            return np.exp(-last_zr * T)

        if T < min(m for m in mats if m > 0):
            # Avant le premier pilier non-nul: interpoler depuis T=0
            first_pos_idx = next(i for i, m in enumerate(mats) if m > 0)
            t2 = mats[first_pos_idx]
            df2 = dfs[first_pos_idx]
            weight = T / t2
            log_df = np.log(df2) * weight  # log(1.0)*(1-w) + log(df2)*w
            return np.exp(log_df)

        # Trouver les piliers encadrants
        idx_lower, idx_upper = self._find_bracket(T)
        t1, t2 = mats[idx_lower], mats[idx_upper]
        df1, df2 = dfs[idx_lower], dfs[idx_upper]

        if self.interpolation == 'linear':
            # Interpolation linéaire sur les taux zéro
            zr1 = self.zero_rates[idx_lower] if t1 > 0 else self.zero_rates[idx_upper]
            zr2 = self.zero_rates[idx_upper]
            weight = (T - t1) / (t2 - t1) if t2 > t1 else 0.0
            zr_interp = zr1 * (1 - weight) + zr2 * weight
            return np.exp(-zr_interp * T)

        elif self.interpolation == 'log_linear':
            # Interpolation log-linéaire sur les DF
            weight = (T - t1) / (t2 - t1) if t2 > t1 else 0.0
            log_df = np.log(df1) * (1 - weight) + np.log(df2) * weight
            return np.exp(log_df)

        elif self.interpolation == 'cubic_spline':
            if self._spline is not None:
                zr_interp = float(self._spline(T))
                return np.exp(-zr_interp * T)
            # Fallback si spline non construite
            weight = (T - t1) / (t2 - t1) if t2 > t1 else 0.0
            zr1 = self.zero_rates[idx_lower] if t1 > 0 else self.zero_rates[idx_upper]
            zr2 = self.zero_rates[idx_upper]
            zr_interp = zr1 * (1 - weight) + zr2 * weight
            return np.exp(-zr_interp * T)

        # Ne devrait jamais arriver
        raise ValueError(f"Interpolation inconnue: {self.interpolation}")

    def zero_rate(self, T: float) -> float:
        """
        Taux zéro (composé continu) à la maturité T.

        Formule: r(T) = -ln(DF(T)) / T

        Paramètres:
        -----------
        T : float
            Maturité en années

        Retourne:
        ---------
        float : taux zéro continu
        """
        if T <= 0:
            # Limite : utiliser le premier taux disponible
            if len(self.zero_rates) > 1:
                return self.zero_rates[1]
            return 0.0
        df = self.discount_factor(T)
        return -np.log(df) / T

    def forward_rate(self, T1: float, T2: float) -> float:
        """
        Taux forward entre T1 et T2.

        Formule: f(T1,T2) = -[ln(DF(T2)) - ln(DF(T1))] / (T2 - T1)

        Paramètres:
        -----------
        T1 : float
            Début de la période
        T2 : float
            Fin de la période

        Retourne:
        ---------
        float : taux forward continu

        Raises:
        -------
        ValueError
            Si T2 <= T1
        """
        if T2 <= T1:
            raise ValueError(f"T2 ({T2}) doit être > T1 ({T1})")
        df1 = self.discount_factor(T1)
        df2 = self.discount_factor(T2)
        return -(np.log(df2) - np.log(df1)) / (T2 - T1)

    def instantaneous_forward(self, T: float, dT: float = 0.001) -> float:
        """
        Taux forward instantané à T.

        Approximation numérique: f(T) ≈ f(T, T+dT)

        Paramètres:
        -----------
        T : float
            Maturité
        dT : float
            Pas pour l'approximation (défaut: 0.001)

        Retourne:
        ---------
        float : taux forward instantané
        """
        return self.forward_rate(max(T, 1e-6), T + dT)

    def get_discount_factors(self, maturities: list[float]) -> np.ndarray:
        """
        Facteurs d'actualisation à des maturités spécifiées.

        Utile pour l'intégration avec IRS.

        Paramètres:
        -----------
        maturities : list[float]
            Liste de maturités

        Retourne:
        ---------
        np.ndarray : tableau de DF
        """
        return np.array([self.discount_factor(t) for t in maturities])

    def get_forward_rates(
        self, payment_times: list[float], payment_frequency: int = 2
    ) -> np.ndarray:
        """
        Taux forward pour chaque période de paiement.

        Pour chaque date t_i, calcule f(t_i - delta, t_i) où delta = 1/freq.

        Paramètres:
        -----------
        payment_times : list[float]
            Dates de paiement
        payment_frequency : int
            Fréquence (pour calculer le début de chaque période)

        Retourne:
        ---------
        np.ndarray : tableau de taux forward
        """
        delta = 1.0 / payment_frequency
        forwards = []
        for t in payment_times:
            t_start = max(t - delta, 0.0)
            fwd = self.forward_rate(t_start, t)
            forwards.append(fwd)
        return np.array(forwards)

    # ------------------------------------------------------------------
    # Analyse de sensibilité
    # ------------------------------------------------------------------

    def dv01(self, bump_bps: float = 1) -> dict:
        """
        DV01 : impact d'un choc parallèle de +1bp sur la courbe.

        Retourne les DF avant et après bump.

        Paramètres:
        -----------
        bump_bps : float
            Taille du bump en points de base (défaut: 1bp)

        Retourne:
        ---------
        dict : {'maturities': [...], 'df_base': [...], 'df_bumped': [...],
                'delta_df': [...]}
        """
        bump = bump_bps / 10_000
        mats = [m for m in self.maturities if m > 0]
        df_base = [self.discount_factor(t) for t in mats]
        df_bumped = [np.exp(-(self.zero_rate(t) + bump) * t) for t in mats]
        delta_df = [b - a for a, b in zip(df_base, df_bumped)]

        return {
            'maturities': mats,
            'df_base': df_base,
            'df_bumped': df_bumped,
            'delta_df': delta_df,
        }

    def key_rate_durations(
        self, pillars: Optional[list[float]] = None, bump_bps: float = 1
    ) -> dict:
        """
        Key rate durations : bump individuel de chaque pilier.

        Paramètres:
        -----------
        pillars : list[float] ou None
            Piliers à bumper (défaut: tous les piliers > 0)
        bump_bps : float
            Taille du bump en points de base

        Retourne:
        ---------
        dict : {pillar_maturity: {'df_change': float, 'zr_change': float}}
        """
        bump = bump_bps / 10_000
        if pillars is None:
            pillars = [m for m in self.maturities if m > 0]

        result = {}
        for pillar in pillars:
            # Trouver l'index du pilier
            idx = None
            for i, m in enumerate(self.maturities):
                if abs(m - pillar) < 1e-10:
                    idx = i
                    break
            if idx is None:
                continue

            # Sauvegarder le DF original
            original_zr = self.zero_rates[idx]
            bumped_zr = original_zr + bump
            df_original = self.discount_factors[idx]
            df_bumped = np.exp(-bumped_zr * pillar)

            result[pillar] = {
                'df_change': df_bumped - df_original,
                'zr_change': bump,
            }

        return result

    # ------------------------------------------------------------------
    # Utilitaires privés
    # ------------------------------------------------------------------

    def _rebuild_zero_rates(self) -> None:
        """Recalcule les taux zéro à partir des DF courants."""
        self.zero_rates = []
        for mat, df in zip(self.maturities, self.discount_factors):
            if mat <= 0:
                self.zero_rates.append(0.0)
            else:
                self.zero_rates.append(-np.log(df) / mat)

    def _rebuild_spline(self) -> None:
        """Reconstruit la spline cubique si nécessaire."""
        if self.interpolation == 'cubic_spline' and len(self.maturities) >= 3:
            valid_idx = [i for i in range(len(self.maturities)) if self.maturities[i] > 0]
            if len(valid_idx) >= 2:
                mats_valid = [self.maturities[i] for i in valid_idx]
                zrs_valid = [self.zero_rates[i] for i in valid_idx]
                self._spline = CubicSpline(mats_valid, zrs_valid, bc_type='natural')
        else:
            self._spline = None

    def _find_bracket(self, T: float) -> tuple[int, int]:
        """Trouve les indices des piliers encadrant T."""
        mats = self.maturities
        idx_lower = 0
        idx_upper = len(mats) - 1

        for i in range(len(mats)):
            if mats[i] <= T:
                idx_lower = i
        for i in range(len(mats)):
            if mats[i] >= T:
                idx_upper = i
                break

        return idx_lower, idx_upper


class NelsonSiegelCurve:
    """
    Courbe des taux paramétrique Nelson-Siegel.

    Modèle (Nelson & Siegel, 1987):
        r(T) = β₀ + β₁ * [(1 - e^(-T/τ)) / (T/τ)]
                   + β₂ * [(1 - e^(-T/τ)) / (T/τ) - e^(-T/τ)]

    Interprétation des paramètres:
        β₀ : niveau long terme (asymptote)
        β₁ : composante de pente (court terme)
        β₂ : composante de courbure (moyen terme)
        τ  : paramètre de décroissance (vitesse de convergence)

    Exemple:
    --------
    >>> ns = NelsonSiegelCurve()
    >>> ns.fit(maturities=[0.5, 1, 2, 5, 10],
    ...        zero_rates=[0.03, 0.032, 0.035, 0.038, 0.04])
    >>> print(f"Taux 3Y = {ns.zero_rate(3.0):.4%}")
    """

    def __init__(self):
        """Initialise avec des paramètres nuls."""
        self.beta0: float = 0.0
        self.beta1: float = 0.0
        self.beta2: float = 0.0
        self.tau: float = 1.0

    @staticmethod
    def _ns_rate(T: float, beta0: float, beta1: float, beta2: float, tau: float) -> float:
        """
        Calcule le taux Nelson-Siegel pour une maturité T.

        Paramètres:
        -----------
        T : float
            Maturité
        beta0, beta1, beta2, tau : float
            Paramètres du modèle

        Retourne:
        ---------
        float : taux zéro continu
        """
        if T < 1e-10:
            # Limite quand T → 0 : r(0) = β₀ + β₁
            return beta0 + beta1

        x = T / tau
        factor1 = (1 - np.exp(-x)) / x
        factor2 = factor1 - np.exp(-x)
        return beta0 + beta1 * factor1 + beta2 * factor2

    def fit(self, maturities: list[float] | np.ndarray, zero_rates: list[float] | np.ndarray) -> None:
        """
        Calibre les paramètres β₀, β₁, β₂, τ sur des taux zéro observés.

        Minimise la somme des carrés des écarts :
            min Σ [r_obs(T_i) - r_NS(T_i; β₀, β₁, β₂, τ)]²

        Paramètres:
        -----------
        maturities : array-like
            Maturités observées
        zero_rates : array-like
            Taux zéro observés
        """
        maturities = np.asarray(maturities, dtype=float)
        zero_rates = np.asarray(zero_rates, dtype=float)

        def objective(params):
            b0, b1, b2, t = params
            if t <= 0.01:
                return 1e10
            fitted = np.array([self._ns_rate(T, b0, b1, b2, t) for T in maturities])
            return np.sum((fitted - zero_rates) ** 2)

        # Initialisation heuristique
        b0_init = zero_rates[-1]  # Taux long terme
        b1_init = zero_rates[0] - zero_rates[-1]  # Pente
        b2_init = 0.0
        tau_init = 2.0

        result = minimize(
            objective,
            x0=[b0_init, b1_init, b2_init, tau_init],
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 10000},
        )

        self.beta0, self.beta1, self.beta2, self.tau = result.x
        logger.info(
            "NS calibre: beta0=%.6f, beta1=%.6f, beta2=%.6f, tau=%.4f (SSE=%.2e)",
            self.beta0, self.beta1, self.beta2, self.tau, result.fun
        )

    def zero_rate(self, T: float) -> float:
        """
        Taux zéro à la maturité T.

        Paramètres:
        -----------
        T : float
            Maturité en années

        Retourne:
        ---------
        float : taux zéro continu r(T)
        """
        return self._ns_rate(T, self.beta0, self.beta1, self.beta2, self.tau)

    def discount_factor(self, T: float) -> float:
        """
        Facteur d'actualisation à la maturité T.

        Formule: DF(T) = exp(-r(T) * T)

        Paramètres:
        -----------
        T : float
            Maturité en années

        Retourne:
        ---------
        float : DF(T)
        """
        if T <= 0:
            return 1.0
        return np.exp(-self.zero_rate(T) * T)


# ======================================================================
# Comparaison flat vs bootstrapped vs Nelson-Siegel
# ======================================================================

def compare_flat_vs_bootstrapped() -> dict:
    """
    Compare le pricing d'un IRS sous trois courbes différentes :
        1. Courbe plate (3.5%)
        2. Courbe bootstrappée (données EUR réalistes, courbe inversée)
        3. Courbe Nelson-Siegel calibrée sur la courbe bootstrappée

    Données EUR approximatives (scénario courbe inversée) :
        - Dépôt 3M : 3.0%
        - Dépôt 6M : 3.2%
        - Swap 1Y : 3.5%
        - Swap 2Y : 3.3%
        - Swap 5Y : 3.1%
        - Swap 10Y : 3.0%

    Retourne:
    ---------
    dict : {'flat': float, 'bootstrapped': float, 'nelson_siegel': float,
            'details': str}
    """
    from .interest_rate import InterestRateSwap

    notional = 10_000_000
    fixed_rate = 0.035
    maturity = 5
    freq = 2

    # --- 1. Courbe plate ---
    swap_flat = InterestRateSwap(notional, fixed_rate, maturity=maturity,
                                  payment_frequency=freq)
    swap_flat._generate_default_curve(0.035)
    price_flat = swap_flat.price()

    # --- 2. Courbe bootstrappée ---
    deposit_rates = [0.03, 0.032]
    deposit_maturities = [0.25, 0.5]
    swap_rates = [0.035, 0.033, 0.031, 0.030]
    swap_maturities_input = [1, 2, 5, 10]

    curve_boot = YieldCurve(interpolation='log_linear')
    curve_boot.build_from_deposits_and_swaps(
        deposit_rates, deposit_maturities,
        swap_rates, swap_maturities_input,
        payment_frequency=freq
    )

    swap_boot = InterestRateSwap(notional, fixed_rate, maturity=maturity,
                                  payment_frequency=freq)
    swap_boot.set_curve(curve_boot)
    price_boot = swap_boot.price()

    # --- 3. Nelson-Siegel ---
    ns_maturities = [m for m in curve_boot.maturities if m > 0]
    ns_rates = [curve_boot.zero_rate(t) for t in ns_maturities]

    ns_curve = NelsonSiegelCurve()
    ns_curve.fit(ns_maturities, ns_rates)

    swap_ns = InterestRateSwap(notional, fixed_rate, maturity=maturity,
                                payment_frequency=freq)
    # Extraire DF et forwards depuis NS
    payment_times = swap_ns.payment_times
    dfs_ns = np.array([ns_curve.discount_factor(t) for t in payment_times])
    fwds_ns = np.array([
        -(np.log(ns_curve.discount_factor(t)) - np.log(ns_curve.discount_factor(max(t - 1/freq, 0)))) / (1/freq)
        for t in payment_times
    ])
    swap_ns.set_yield_curve(dfs_ns, fwds_ns)
    price_ns = swap_ns.price()

    details = (
        f"IRS 5Y, fixe={fixed_rate:.2%}, notional={notional:,.0f} EUR\n"
        f"  Courbe plate (3.5%):       V = {price_flat:>12,.2f} EUR\n"
        f"  Courbe bootstrappee:       V = {price_boot:>12,.2f} EUR\n"
        f"  Nelson-Siegel:             V = {price_ns:>12,.2f} EUR\n"
        f"  Ecart flat vs bootstrap:   {abs(price_boot - price_flat):>12,.2f} EUR\n"
        f"\n"
        f"Impact sur le risk management :\n"
        f"  La courbe inversee implique des taux forward plus bas a long terme,\n"
        f"  ce qui modifie la valeur des jambes fixes et variables du swap.\n"
        f"  Utiliser une courbe plate surestime/sous-estime systematiquement\n"
        f"  les sensibilites (DV01, key rate durations)."
    )

    return {
        'flat': price_flat,
        'bootstrapped': price_boot,
        'nelson_siegel': price_ns,
        'details': details,
    }

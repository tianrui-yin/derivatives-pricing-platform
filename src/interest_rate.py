# -*- coding: utf-8 -*-
"""
Produits de Taux d'Intérêt
==========================
Swaps de taux (IRS), Caps et Floors.

Référence: Hull, Chapitres 7 (Swaps) et 29 (Caps/Floors)

Produits implémentés:
    - Interest Rate Swap (IRS): échange taux fixe vs variable
    - Cap: protection contre hausse des taux
    - Floor: protection contre baisse des taux
"""

import numpy as np
from scipy.stats import norm
from .base_product import BaseInterestRateProduct, BaseDerivative
from .utils import black_price


class InterestRateSwap(BaseInterestRateProduct):
    """
    Swap de taux d'intérêt (IRS).

    Un IRS est un échange de flux:
        - Jambe fixe: paiements fixes périodiques
        - Jambe variable: paiements indexés sur un taux (ex: EURIBOR)

    Valeur pour le payeur du fixe:
        V = PV(jambe variable) - PV(jambe fixe)

    Référence: Hull, Chapitre 7

    Exemple:
    --------
    >>> swap = InterestRateSwap(notional=10000000, fixed_rate=0.02,
    ...                          floating_spread=0, maturity=5)
    >>> swap.set_yield_curve(discount_factors)
    >>> print(f"Valeur: {swap.price():,.0f} €")
    """

    def __init__(self, notional, fixed_rate, floating_spread=0, maturity=5,
                 payment_frequency=2):
        """
        Initialise un swap de taux.

        Paramètres:
        -----------
        notional : float
            Nominal du swap
        fixed_rate : float
            Taux fixe du swap (ex: 0.02 pour 2%)
        floating_spread : float
            Spread sur le taux variable (ex: 0.001 pour 10bp)
        maturity : float
            Maturité en années
        payment_frequency : int
            Fréquence de paiement par an (2 = semestriel)
        """
        super().__init__(notional, fixed_rate, 0, maturity,
                        name="Interest Rate Swap")
        self.floating_spread = floating_spread
        self.maturity = maturity
        self.payment_frequency = payment_frequency

        # Calculer les dates de paiement
        self.n_payments = int(maturity * payment_frequency)
        self.payment_times = np.array([(i + 1) / payment_frequency
                                       for i in range(self.n_payments)])

        # Courbe des taux (à définir)
        self.discount_factors = None
        self.forward_rates = None

    def set_yield_curve(self, discount_factors=None, forward_rates=None):
        """
        Définit la courbe des taux pour le pricing.

        Paramètres:
        -----------
        discount_factors : array-like
            Facteurs d'actualisation pour chaque date de paiement
        forward_rates : array-like
            Taux forward pour chaque période
        """
        if discount_factors is not None:
            self.discount_factors = np.array(discount_factors)

        if forward_rates is not None:
            self.forward_rates = np.array(forward_rates)

    def _generate_default_curve(self, flat_rate=0.025):
        """
        Génère une courbe plate pour les tests.

        Paramètres:
        -----------
        flat_rate : float
            Taux de la courbe plate
        """
        self.discount_factors = np.exp(-flat_rate * self.payment_times)
        self.forward_rates = np.full(self.n_payments, flat_rate)

    def price(self):
        """
        Calcule la valeur du swap (du point de vue du payeur du fixe).

        Méthode (Hull, Section 7.7):
            V = B_float - B_fixed

        où B_float et B_fixed sont les valeurs des deux jambes.

        Pour un swap à la création:
            - Jambe variable ≈ notional (par définition)
            - Valeur ≈ 0 si taux fixe = taux swap de marché

        Retourne:
        ---------
        float : valeur du swap
        """
        if self.discount_factors is None:
            self._generate_default_curve()

        # Fraction d'année entre paiements
        delta = 1 / self.payment_frequency

        # Valeur de la jambe fixe (coupons fixes + principal à maturité)
        # PV(fixe) = Σ(c * delta * DF_i) + N * DF_n
        fixed_coupons = self.fixed_rate * delta * self.notional * np.sum(self.discount_factors)
        fixed_leg = fixed_coupons + self.notional * self.discount_factors[-1]

        # Valeur de la jambe variable
        # À l'initiation d'un swap, la jambe variable vaut le notional
        # car les taux forward sont cohérents avec les facteurs d'actualisation
        # PV(variable) = N (au début)
        floating_leg = self.notional

        # Ajouter le spread si présent
        if self.floating_spread != 0:
            spread_leg = self.floating_spread * delta * self.notional * \
                        np.sum(self.discount_factors)
            floating_leg += spread_leg

        # Valeur pour le payeur du fixe = jambe variable - jambe fixe
        swap_value = floating_leg - fixed_leg

        return swap_value

    def dv01(self, bump=0.0001):
        """
        Calcule le DV01 (Dollar Value of One Basis Point).

        Le DV01 mesure la sensibilité du swap à un mouvement
        de 1 point de base (0.01%) des taux.

        Méthode: Bump des facteurs d'actualisation et recalcul.

        Paramètres:
        -----------
        bump : float
            Taille du bump (défaut: 1bp = 0.0001)

        Retourne:
        ---------
        float : DV01 (changement de valeur pour +1bp)
        """
        if self.discount_factors is None:
            self._generate_default_curve()

        # Valeur actuelle
        value_base = self.price()

        # Bumper les facteurs d'actualisation (taux + 1bp)
        df_bumped = self.discount_factors * np.exp(-bump * self.payment_times)
        df_original = self.discount_factors.copy()

        self.discount_factors = df_bumped
        value_bumped = self.price()

        # Restaurer
        self.discount_factors = df_original

        # DV01 = changement de valeur
        dv01 = value_base - value_bumped

        return dv01

    def par_rate(self):
        """
        Calcule le taux swap (par rate).

        Le taux swap est le taux fixe qui rend la valeur du swap = 0.

        Formule (Hull, Eq. 7.5):
            r_swap = (1 - DF_n) / Σ(delta_i * DF_i)

        Retourne:
        ---------
        float : taux swap
        """
        if self.discount_factors is None:
            self._generate_default_curve()

        delta = 1 / self.payment_frequency
        sum_df = np.sum(self.discount_factors) * delta

        par_rate = (1 - self.discount_factors[-1]) / sum_df

        return par_rate

    def description(self):
        """
        Description du swap.
        """
        return (f"IRS: Notional={self.notional:,.0f}, "
                f"Fixe={self.fixed_rate:.2%}, "
                f"Maturité={self.maturity}ans, "
                f"Fréq={self.payment_frequency}x/an")


class Cap(BaseDerivative):
    """
    Cap sur taux d'intérêt.

    Un Cap est un ensemble de caplets qui protègent contre
    la hausse des taux au-delà d'un strike.

    Chaque caplet paie: max(L - K, 0) * delta * notional
    où L est le taux LIBOR/EURIBOR et K est le strike.

    Pricing: Modèle de Black (Hull, Chapitre 29)

    Exemple:
    --------
    >>> cap = Cap(notional=10000000, strike=0.025, maturity=3, vol=0.15)
    >>> print(f"Prix: {cap.price():,.0f} €")
    """

    def __init__(self, notional, strike, maturity, vol, payment_frequency=2):
        """
        Initialise un Cap.

        Paramètres:
        -----------
        notional : float
            Nominal
        strike : float
            Strike du cap (taux maximum)
        maturity : float
            Maturité
        vol : float
            Volatilité (pour le modèle de Black)
        payment_frequency : int
            Fréquence de paiement par an
        """
        super().__init__(name="Interest Rate Cap", notional=notional)
        self.strike = strike
        self.maturity = maturity
        self.vol = vol
        self.payment_frequency = payment_frequency

        # Dates des caplets
        self.n_caplets = int(maturity * payment_frequency) - 1
        self.fixing_times = np.array([(i + 1) / payment_frequency
                                      for i in range(self.n_caplets)])
        self.payment_times = self.fixing_times + 1 / payment_frequency

        # Courbe des taux
        self.discount_factors = None
        self.forward_rates = None

    def set_market_data(self, discount_factors, forward_rates):
        """
        Définit les données de marché.

        Paramètres:
        -----------
        discount_factors : array-like
            Facteurs d'actualisation aux dates de paiement
        forward_rates : array-like
            Taux forward pour chaque caplet
        """
        self.discount_factors = np.array(discount_factors)
        self.forward_rates = np.array(forward_rates)

    def _generate_default_market(self, flat_rate=0.02):
        """
        Génère des données de marché par défaut.
        """
        self.discount_factors = np.exp(-flat_rate * self.payment_times)
        self.forward_rates = np.full(self.n_caplets, flat_rate)

    def price(self):
        """
        Calcule le prix du Cap (somme des caplets).

        Chaque caplet est pricé avec le modèle de Black:
            Caplet = delta * DF * [F*N(d1) - K*N(d2)]

        Retourne:
        ---------
        float : prix du cap
        """
        if self.discount_factors is None:
            self._generate_default_market()

        delta = 1 / self.payment_frequency
        total_price = 0

        for i in range(self.n_caplets):
            F = self.forward_rates[i]  # Taux forward
            T = self.fixing_times[i]   # Date de fixing
            DF = self.discount_factors[i]

            # Prix du caplet (formule de Black)
            if T > 0:
                d1 = (np.log(F / self.strike) + 0.5 * self.vol**2 * T) / \
                     (self.vol * np.sqrt(T))
                d2 = d1 - self.vol * np.sqrt(T)

                caplet_price = delta * self.notional * DF * \
                              (F * norm.cdf(d1) - self.strike * norm.cdf(d2))
            else:
                caplet_price = delta * self.notional * DF * \
                              max(F - self.strike, 0)

            total_price += caplet_price

        return total_price

    def description(self):
        """
        Description du Cap.
        """
        return (f"Cap: Notional={self.notional:,.0f}, "
                f"Strike={self.strike:.2%}, "
                f"Maturité={self.maturity}ans, "
                f"Vol={self.vol:.1%}")


class Floor(BaseDerivative):
    """
    Floor sur taux d'intérêt.

    Un Floor protège contre la baisse des taux en dessous d'un strike.

    Chaque floorlet paie: max(K - L, 0) * delta * notional

    Propriété (Cap-Floor Parity):
        Cap - Floor = Swap (payeur fixe)

    Exemple:
    --------
    >>> floor = Floor(notional=10000000, strike=0.01, maturity=3, vol=0.15)
    >>> print(f"Prix: {floor.price():,.0f} €")
    """

    def __init__(self, notional, strike, maturity, vol, payment_frequency=2):
        """
        Initialise un Floor.

        Paramètres: identiques au Cap
        """
        super().__init__(name="Interest Rate Floor", notional=notional)
        self.strike = strike
        self.maturity = maturity
        self.vol = vol
        self.payment_frequency = payment_frequency

        self.n_floorlets = int(maturity * payment_frequency) - 1
        self.fixing_times = np.array([(i + 1) / payment_frequency
                                      for i in range(self.n_floorlets)])
        self.payment_times = self.fixing_times + 1 / payment_frequency

        self.discount_factors = None
        self.forward_rates = None

    def set_market_data(self, discount_factors, forward_rates):
        """
        Définit les données de marché.
        """
        self.discount_factors = np.array(discount_factors)
        self.forward_rates = np.array(forward_rates)

    def _generate_default_market(self, flat_rate=0.02):
        """
        Génère des données de marché par défaut.
        """
        self.discount_factors = np.exp(-flat_rate * self.payment_times)
        self.forward_rates = np.full(self.n_floorlets, flat_rate)

    def price(self):
        """
        Calcule le prix du Floor (somme des floorlets).

        Chaque floorlet est pricé avec le modèle de Black (put):
            Floorlet = delta * DF * [K*N(-d2) - F*N(-d1)]

        Retourne:
        ---------
        float : prix du floor
        """
        if self.discount_factors is None:
            self._generate_default_market()

        delta = 1 / self.payment_frequency
        total_price = 0

        for i in range(self.n_floorlets):
            F = self.forward_rates[i]
            T = self.fixing_times[i]
            DF = self.discount_factors[i]

            if T > 0:
                d1 = (np.log(F / self.strike) + 0.5 * self.vol**2 * T) / \
                     (self.vol * np.sqrt(T))
                d2 = d1 - self.vol * np.sqrt(T)

                floorlet_price = delta * self.notional * DF * \
                                (self.strike * norm.cdf(-d2) - F * norm.cdf(-d1))
            else:
                floorlet_price = delta * self.notional * DF * \
                                max(self.strike - F, 0)

            total_price += floorlet_price

        return total_price

    def description(self):
        """
        Description du Floor.
        """
        return (f"Floor: Notional={self.notional:,.0f}, "
                f"Strike={self.strike:.2%}, "
                f"Maturité={self.maturity}ans, "
                f"Vol={self.vol:.1%}")


# Tests si exécuté directement
if __name__ == "__main__":
    print("=== Test Produits de Taux ===")

    # Paramètres
    notional = 10_000_000  # 10 millions €
    maturity = 5
    flat_rate = 0.025  # 2.5%

    # Interest Rate Swap
    print(f"\n--- Interest Rate Swap ---")
    swap = InterestRateSwap(notional, fixed_rate=0.025, maturity=maturity)
    swap._generate_default_curve(flat_rate)

    print(f"Notional: {notional:,.0f} €")
    print(f"Taux fixe: {swap.fixed_rate:.2%}")
    print(f"Valeur du swap: {swap.price():,.0f} €")
    print(f"Taux swap (par rate): {swap.par_rate():.4%}")
    print(f"DV01: {swap.dv01():,.0f} €")

    # Cap
    print(f"\n--- Cap ---")
    cap = Cap(notional, strike=0.03, maturity=maturity, vol=0.20)
    cap._generate_default_market(flat_rate)
    print(f"Strike: {cap.strike:.2%}")
    print(f"Prix du Cap: {cap.price():,.0f} €")

    # Floor
    print(f"\n--- Floor ---")
    floor = Floor(notional, strike=0.02, maturity=maturity, vol=0.20)
    floor._generate_default_market(flat_rate)
    print(f"Strike: {floor.strike:.2%}")
    print(f"Prix du Floor: {floor.price():,.0f} €")

    # Vérification Cap-Floor Parity
    print(f"\n--- Vérification Cap-Floor Parity ---")
    cap_atm = Cap(notional, strike=flat_rate, maturity=maturity, vol=0.20)
    floor_atm = Floor(notional, strike=flat_rate, maturity=maturity, vol=0.20)
    cap_atm._generate_default_market(flat_rate)
    floor_atm._generate_default_market(flat_rate)

    print(f"Cap ATM: {cap_atm.price():,.0f} €")
    print(f"Floor ATM: {floor_atm.price():,.0f} €")
    print(f"Cap - Floor: {cap_atm.price() - floor_atm.price():,.0f} € (≈ 0 pour ATM)")

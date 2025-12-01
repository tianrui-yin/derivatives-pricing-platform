# -*- coding: utf-8 -*-
"""
Options Lookback
================
Options dont le payoff dépend du minimum ou maximum atteint.

Référence: Hull, Chapitre 26 - Exotic Options (Section 26.5)

Types:
    - Floating Strike: le strike est le min/max
        - Call: S_T - S_min (le droit d'acheter au plus bas)
        - Put: S_max - S_T (le droit de vendre au plus haut)

    - Fixed Strike: strike fixé, payoff sur min/max
        - Call: max(S_max - K, 0)
        - Put: max(K - S_min, 0)

Ces options sont plus chères que les vanilles car elles éliminent
le risque de mal "timer" le marché.
"""

import numpy as np
from scipy.stats import norm
from .base_product import BaseOption
from .utils import simulate_gbm_paths


class LookbackOption(BaseOption):
    """
    Option lookback.

    Pricing: Monte Carlo ou formules analytiques.

    Exemple:
    --------
    >>> lookback = LookbackOption(S=100, K=100, T=1, r=0.05, sigma=0.2,
    ...                            lookback_type='floating', option_type='call')
    >>> print(f"Prix: {lookback.price():.4f}")
    """

    def __init__(self, S, K, T, r, sigma, lookback_type='floating',
                 option_type='call'):
        """
        Initialise une option lookback.

        Paramètres:
        -----------
        S : float
            Prix spot (aussi utilisé comme S_min/S_max initial)
        K : float
            Strike (pour fixed strike lookback)
        T : float
            Maturité
        r : float
            Taux sans risque
        sigma : float
            Volatilité
        lookback_type : str
            'floating' (strike = min/max) ou 'fixed' (strike fixé)
        option_type : str
            'call' ou 'put'
        """
        super().__init__(S, K, T, r, sigma, option_type,
                        name=f"Lookback Option ({lookback_type})")
        self.lookback_type = lookback_type.lower()

        if self.lookback_type not in ['floating', 'fixed']:
            raise ValueError("lookback_type doit être 'floating' ou 'fixed'")

    def price(self, method='monte_carlo', n_paths=50000, n_steps=252):
        """
        Calcule le prix de l'option lookback.

        Paramètres:
        -----------
        method : str
            'monte_carlo' ou 'analytical'
        n_paths : int
            Nombre de simulations
        n_steps : int
            Nombre de pas

        Retourne:
        ---------
        float : prix de l'option
        """
        if method == 'analytical' and self.lookback_type == 'floating':
            return self._price_floating_analytical()
        else:
            return self._price_monte_carlo(n_paths, n_steps)

    def _price_floating_analytical(self):
        """
        Formule analytique pour le lookback floating strike.

        Formules de Goldman-Sosin-Gatto (Hull, Section 26.5).

        Pour un call floating strike (payoff = S_T - S_min):
            c = S*N(a1) - S*e^(-rT)*σ²/(2r)*N(-a1) - S*e^(-rT)*N(a2)
                + S*e^(-rT)*σ²/(2r)*[S/S_min]^(-2r/σ²)*N(a3)

        Où S_min = S (au début, le minimum est le spot actuel)
        """
        # Simplification: au temps 0, S_min = S_max = S
        a1 = (np.log(1) + (self.r + self.sigma**2 / 2) * self.T) / \
             (self.sigma * np.sqrt(self.T))  # log(S/S_min) = 0
        a2 = a1 - self.sigma * np.sqrt(self.T)
        a3 = a1 - 2 * self.r * np.sqrt(self.T) / self.sigma

        discount = np.exp(-self.r * self.T)
        term1 = self.sigma**2 / (2 * self.r)

        if self.option_type == 'call':
            # Floating strike call: payoff = S_T - S_min
            price = (self.S * norm.cdf(a1) -
                    self.S * discount * term1 * norm.cdf(-a1) -
                    self.S * discount * norm.cdf(a2) +
                    self.S * discount * term1 * norm.cdf(a3))
        else:
            # Floating strike put: payoff = S_max - S_T
            # Formule symétrique
            price = (-self.S * norm.cdf(-a1) +
                    self.S * discount * term1 * norm.cdf(a1) +
                    self.S * discount * norm.cdf(-a2) -
                    self.S * discount * term1 * norm.cdf(-a3))

        return max(price, 0)

    def _price_monte_carlo(self, n_paths=50000, n_steps=252):
        """
        Pricing Monte Carlo pour les options lookback.

        Méthode:
            1. Simuler des chemins de prix
            2. Calculer min/max sur chaque chemin
            3. Calculer le payoff selon le type
            4. Actualiser et moyenner

        Retourne:
        ---------
        float : prix Monte Carlo
        """
        # Simuler les chemins
        paths = simulate_gbm_paths(self.S, self.r, self.sigma, self.T,
                                   n_steps, n_paths)

        # Prix final
        final_prices = paths[:, -1]

        # Minimum et maximum sur chaque chemin
        min_prices = np.min(paths, axis=1)
        max_prices = np.max(paths, axis=1)

        # Calculer les payoffs selon le type
        if self.lookback_type == 'floating':
            if self.option_type == 'call':
                # Floating strike call: payoff = S_T - S_min
                # (le droit d'acheter au prix minimum)
                payoffs = final_prices - min_prices
            else:
                # Floating strike put: payoff = S_max - S_T
                # (le droit de vendre au prix maximum)
                payoffs = max_prices - final_prices
        else:  # fixed strike
            if self.option_type == 'call':
                # Fixed strike call: payoff = max(S_max - K, 0)
                payoffs = np.maximum(max_prices - self.K, 0)
            else:
                # Fixed strike put: payoff = max(K - S_min, 0)
                payoffs = np.maximum(self.K - min_prices, 0)

        # Prix = espérance actualisée
        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)

        return price

    def lookback_premium(self):
        """
        Calcule la prime du lookback par rapport au vanille.

        La prime est toujours positive car le lookback élimine
        le risque de mauvais timing.

        Retourne:
        ---------
        float : ratio prix lookback / prix vanille
        """
        from .vanilla_options import EuropeanOption

        lookback_price = self.price()

        if self.lookback_type == 'floating':
            # Pour floating, comparer avec ATM forward
            vanilla = EuropeanOption(self.S, self.S, self.T, self.r, self.sigma,
                                     self.option_type)
        else:
            vanilla = EuropeanOption(self.S, self.K, self.T, self.r, self.sigma,
                                     self.option_type)

        vanilla_price = vanilla.price()

        if vanilla_price > 0:
            return lookback_price / vanilla_price
        return 1.0

    def description(self):
        """
        Description de l'option lookback.
        """
        return (f"Lookback {self.option_type.upper()} ({self.lookback_type}): "
                f"S={self.S}, K={self.K}, T={self.T:.2f}ans, σ={self.sigma:.1%}")


def compare_lookback_types(S, K, T, r, sigma):
    """
    Compare les différents types d'options lookback.

    Paramètres:
    -----------
    S, K, T, r, sigma : paramètres standard

    Retourne:
    ---------
    dict : comparaison des prix
    """
    from .vanilla_options import EuropeanOption

    # Vanilles
    vanilla_call = EuropeanOption(S, K, T, r, sigma, 'call').price()
    vanilla_put = EuropeanOption(S, K, T, r, sigma, 'put').price()

    # Lookbacks
    lb_float_call = LookbackOption(S, K, T, r, sigma, 'floating', 'call').price()
    lb_float_put = LookbackOption(S, K, T, r, sigma, 'floating', 'put').price()
    lb_fixed_call = LookbackOption(S, K, T, r, sigma, 'fixed', 'call').price()
    lb_fixed_put = LookbackOption(S, K, T, r, sigma, 'fixed', 'put').price()

    return {
        'vanilla_call': vanilla_call,
        'vanilla_put': vanilla_put,
        'floating_call': lb_float_call,
        'floating_put': lb_float_put,
        'fixed_call': lb_fixed_call,
        'fixed_put': lb_fixed_put,
        'premium_float_call': lb_float_call / vanilla_call if vanilla_call > 0 else 0,
        'premium_fixed_call': lb_fixed_call / vanilla_call if vanilla_call > 0 else 0
    }


# Tests si exécuté directement
if __name__ == "__main__":
    print("=== Test Options Lookback ===")

    # Paramètres de test
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.25

    print(f"\nParamètres: S={S}, K={K}, T={T}, r={r}, σ={sigma}")

    # Floating strike
    print(f"\n--- Floating Strike Lookback ---")
    lb_float_call = LookbackOption(S, K, T, r, sigma, 'floating', 'call')
    lb_float_put = LookbackOption(S, K, T, r, sigma, 'floating', 'put')

    print(f"Call (S_T - S_min): {lb_float_call.price():.4f} €")
    print(f"Put (S_max - S_T):  {lb_float_put.price():.4f} €")

    # Fixed strike
    print(f"\n--- Fixed Strike Lookback ---")
    lb_fixed_call = LookbackOption(S, K, T, r, sigma, 'fixed', 'call')
    lb_fixed_put = LookbackOption(S, K, T, r, sigma, 'fixed', 'put')

    print(f"Call (S_max - K)+: {lb_fixed_call.price():.4f} €")
    print(f"Put (K - S_min)+:  {lb_fixed_put.price():.4f} €")

    # Comparaison avec vanilles
    print(f"\n--- Comparaison avec Vanilles ---")
    comparison = compare_lookback_types(S, K, T, r, sigma)
    print(f"Vanille Call:      {comparison['vanilla_call']:.4f} €")
    print(f"Floating Call:     {comparison['floating_call']:.4f} € ({comparison['premium_float_call']:.1f}x)")
    print(f"Fixed Call:        {comparison['fixed_call']:.4f} € ({comparison['premium_fixed_call']:.1f}x)")

    # Effet de la volatilité
    print(f"\n--- Effet de la volatilité ---")
    for vol in [0.15, 0.25, 0.35]:
        lb = LookbackOption(S, K, T, r, vol, 'floating', 'call')
        print(f"σ = {vol:.0%}: {lb.price():.4f} € (premium: {lb.lookback_premium():.1f}x)")

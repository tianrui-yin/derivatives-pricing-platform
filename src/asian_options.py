# -*- coding: utf-8 -*-
"""
Options Asiatiques
==================
Options dont le payoff dépend de la moyenne du sous-jacent.

Référence: Hull, Chapitre 26 - Exotic Options (Section 26.6)

Types de moyennes:
    - Arithmétique: (S₁ + S₂ + ... + Sₙ) / n
    - Géométrique: (S₁ × S₂ × ... × Sₙ)^(1/n)

Types de payoff:
    - Average Price: payoff basé sur moyenne vs strike
    - Average Strike: payoff basé sur spot final vs moyenne (non implémenté ici)

Les options asiatiques sont moins chères que les vanilles car
la moyenne réduit la volatilité effective.
"""

import numpy as np
from scipy.stats import norm
from .base_product import BaseOption
from .utils import simulate_gbm_paths


class AsianOption(BaseOption):
    """
    Option asiatique sur la moyenne du sous-jacent.

    Pricing:
        - Géométrique: formule analytique (approximation)
        - Arithmétique: Monte Carlo (pas de formule fermée)

    Exemple:
    --------
    >>> asian = AsianOption(S=100, K=100, T=1, r=0.05, sigma=0.2,
    ...                      average_type='arithmetic', option_type='call')
    >>> print(f"Prix: {asian.price():.4f}")
    """

    def __init__(self, S, K, T, r, sigma, average_type='arithmetic',
                 option_type='call', n_fixings=12):
        """
        Initialise une option asiatique.

        Paramètres:
        -----------
        S, K, T, r, sigma : paramètres standard
        average_type : str
            'arithmetic' ou 'geometric'
        option_type : str
            'call' ou 'put'
        n_fixings : int
            Nombre de fixings pour la moyenne (défaut: 12 = mensuel sur 1 an)
        """
        super().__init__(S, K, T, r, sigma, option_type,
                        name=f"Asian Option ({average_type})")
        self.average_type = average_type.lower()
        self.n_fixings = n_fixings

        if self.average_type not in ['arithmetic', 'geometric']:
            raise ValueError("average_type doit être 'arithmetic' ou 'geometric'")

    def price(self, n_paths=50000):
        """
        Calcule le prix de l'option asiatique.

        Paramètres:
        -----------
        n_paths : int
            Nombre de simulations (pour Monte Carlo)

        Retourne:
        ---------
        float : prix de l'option
        """
        if self.average_type == 'geometric':
            return self._price_geometric_analytical()
        else:
            return self._price_monte_carlo(n_paths)

    def _price_geometric_analytical(self):
        """
        Formule analytique pour l'option asiatique géométrique.

        Propriété: La moyenne géométrique d'un GBM est aussi un GBM,
        ce qui permet d'utiliser une formule de type Black-Scholes.

        Formule (Hull, Section 26.6):
            - Volatilité ajustée: σ_a = σ / √3
            - Drift ajusté: (r - q - σ²/6) / 2

        Pour simplifier, on utilise ici une approximation.
        """
        # Volatilité effective (réduite car moyenne)
        sigma_adj = self.sigma / np.sqrt(3)

        # Drift ajusté
        drift_adj = 0.5 * (self.r - self.sigma**2 / 6)

        # Prix "forward" de la moyenne géométrique
        # Approximation: F_A ≈ S * exp(drift_adj * T)
        F_A = self.S * np.exp(drift_adj * self.T)

        # Utiliser Black-Scholes avec paramètres ajustés
        d_1 = (np.log(F_A / self.K) + 0.5 * sigma_adj**2 * self.T) / \
              (sigma_adj * np.sqrt(self.T))
        d_2 = d_1 - sigma_adj * np.sqrt(self.T)

        discount = np.exp(-self.r * self.T)

        if self.option_type == 'call':
            price = discount * (F_A * norm.cdf(d_1) - self.K * norm.cdf(d_2))
        else:
            price = discount * (self.K * norm.cdf(-d_2) - F_A * norm.cdf(-d_1))

        return max(price, 0)

    def _price_monte_carlo(self, n_paths=50000):
        """
        Pricing Monte Carlo pour l'option asiatique arithmétique.

        Méthode:
            1. Simuler n_paths chemins de prix
            2. Pour chaque chemin, calculer la moyenne (arithmétique)
            3. Calculer le payoff basé sur la moyenne
            4. Actualiser et moyenner

        Retourne:
        ---------
        float : prix Monte Carlo
        """
        # Simuler les chemins aux dates de fixing
        paths = simulate_gbm_paths(self.S, self.r, self.sigma, self.T,
                                   self.n_fixings, n_paths)

        # Calculer la moyenne pour chaque chemin
        if self.average_type == 'arithmetic':
            averages = np.mean(paths[:, 1:], axis=1)  # Exclure le prix initial
        else:  # geometric
            # Moyenne géométrique = exp(moyenne des log)
            averages = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))

        # Calculer les payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(averages - self.K, 0)
        else:
            payoffs = np.maximum(self.K - averages, 0)

        # Prix = espérance actualisée
        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)

        return price

    def asian_discount(self):
        """
        Calcule le rabais par rapport à l'option vanille.

        L'option asiatique est moins chère car la volatilité
        de la moyenne est plus faible.

        Retourne:
        ---------
        float : ratio prix asiatique / prix vanille
        """
        from .vanilla_options import EuropeanOption

        vanilla = EuropeanOption(self.S, self.K, self.T, self.r, self.sigma,
                                 self.option_type)
        vanilla_price = vanilla.price()
        asian_price = self.price()

        if vanilla_price > 0:
            return asian_price / vanilla_price
        return 1.0

    def description(self):
        """
        Description de l'option asiatique.
        """
        return (f"Asian {self.option_type.upper()} ({self.average_type}): "
                f"S={self.S}, K={self.K}, T={self.T:.2f}ans, "
                f"σ={self.sigma:.1%}, {self.n_fixings} fixings")


def compare_average_types(S, K, T, r, sigma, option_type='call'):
    """
    Compare les prix des options asiatiques arithmétique et géométrique.

    La moyenne géométrique est toujours ≤ moyenne arithmétique,
    donc le call géométrique est moins cher que l'arithmétique.

    Paramètres:
    -----------
    S, K, T, r, sigma : paramètres standard
    option_type : str
        'call' ou 'put'

    Retourne:
    ---------
    dict : comparaison des prix
    """
    from .vanilla_options import EuropeanOption

    vanilla = EuropeanOption(S, K, T, r, sigma, option_type)
    asian_arith = AsianOption(S, K, T, r, sigma, 'arithmetic', option_type)
    asian_geom = AsianOption(S, K, T, r, sigma, 'geometric', option_type)

    vanilla_price = vanilla.price()
    arith_price = asian_arith.price()
    geom_price = asian_geom.price()

    return {
        'vanilla': vanilla_price,
        'arithmetic': arith_price,
        'geometric': geom_price,
        'discount_arith': arith_price / vanilla_price if vanilla_price > 0 else 0,
        'discount_geom': geom_price / vanilla_price if vanilla_price > 0 else 0
    }


# Tests si exécuté directement
if __name__ == "__main__":
    print("=== Test Options Asiatiques ===")

    # Paramètres de test
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.25

    # Options asiatiques
    asian_arith_call = AsianOption(S, K, T, r, sigma, 'arithmetic', 'call')
    asian_geom_call = AsianOption(S, K, T, r, sigma, 'geometric', 'call')

    print(f"\nParamètres: S={S}, K={K}, T={T}, r={r}, σ={sigma}")

    print(f"\n--- Options Asiatiques Call ---")
    print(f"Arithmétique (MC): {asian_arith_call.price():.4f} €")
    print(f"Géométrique (ana): {asian_geom_call.price():.4f} €")

    # Comparaison avec vanille
    print(f"\n--- Comparaison avec Vanille ---")
    comparison = compare_average_types(S, K, T, r, sigma, 'call')
    print(f"Vanille:      {comparison['vanilla']:.4f} €")
    print(f"Arithmétique: {comparison['arithmetic']:.4f} € ({comparison['discount_arith']:.1%} de vanille)")
    print(f"Géométrique:  {comparison['geometric']:.4f} € ({comparison['discount_geom']:.1%} de vanille)")

    # Options Put
    asian_arith_put = AsianOption(S, K, T, r, sigma, 'arithmetic', 'put')
    print(f"\n--- Options Asiatiques Put ---")
    print(f"Arithmétique (MC): {asian_arith_put.price():.4f} €")

    # Effet du nombre de fixings
    print(f"\n--- Effet du nombre de fixings ---")
    for n_fix in [4, 12, 52, 252]:
        asian = AsianOption(S, K, T, r, sigma, 'arithmetic', 'call', n_fixings=n_fix)
        print(f"{n_fix:3d} fixings: {asian.price():.4f} €")

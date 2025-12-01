# -*- coding: utf-8 -*-
"""
Options Barrières
=================
Options avec activation ou désactivation selon un niveau de prix.

Référence: Hull, Chapitre 26 - Exotic Options

Types de barrières:
    - Knock-out: l'option devient sans valeur si la barrière est touchée
    - Knock-in: l'option n'est active que si la barrière est touchée

Directions:
    - Down: barrière inférieure au prix initial
    - Up: barrière supérieure au prix initial

Combinaisons: Down-and-Out, Down-and-In, Up-and-Out, Up-and-In
"""

import numpy as np
from scipy.stats import norm
from .base_product import BaseOption
from .utils import simulate_gbm_paths, d1, d2


class BarrierOption(BaseOption):
    """
    Option barrière.

    Pricing: Monte Carlo ou formules analytiques selon le type.

    Exemple:
    --------
    >>> barrier_opt = BarrierOption(S=100, K=100, T=1, r=0.05, sigma=0.2,
    ...                              barrier=80, barrier_type='down-and-out')
    >>> print(f"Prix: {barrier_opt.price():.4f}")
    """

    def __init__(self, S, K, T, r, sigma, barrier, barrier_type='down-and-out',
                 option_type='call'):
        """
        Initialise une option barrière.

        Paramètres:
        -----------
        S, K, T, r, sigma : paramètres standard
        barrier : float
            Niveau de la barrière
        barrier_type : str
            Type de barrière:
            - 'down-and-out': désactivée si S descend sous barrier
            - 'down-and-in': activée si S descend sous barrier
            - 'up-and-out': désactivée si S monte au-dessus de barrier
            - 'up-and-in': activée si S monte au-dessus de barrier
        option_type : str
            'call' ou 'put'
        """
        super().__init__(S, K, T, r, sigma, option_type,
                        name=f"Barrier Option ({barrier_type})")
        self.barrier = barrier
        self.barrier_type = barrier_type.lower()

        # Validation du type de barrière
        valid_types = ['down-and-out', 'down-and-in', 'up-and-out', 'up-and-in']
        if self.barrier_type not in valid_types:
            raise ValueError(f"barrier_type doit être parmi: {valid_types}")

        # Validation de la cohérence barrière/spot
        if 'down' in self.barrier_type and self.barrier >= self.S:
            raise ValueError("Pour 'down', la barrière doit être < S")
        if 'up' in self.barrier_type and self.barrier <= self.S:
            raise ValueError("Pour 'up', la barrière doit être > S")

    def price(self, method='monte_carlo', n_paths=50000, n_steps=252):
        """
        Calcule le prix de l'option barrière.

        Paramètres:
        -----------
        method : str
            'monte_carlo' ou 'analytical' (pour certains cas)
        n_paths : int
            Nombre de simulations Monte Carlo
        n_steps : int
            Nombre de pas par simulation

        Retourne:
        ---------
        float : prix de l'option
        """
        if method == 'analytical':
            return self._price_analytical()
        else:
            return self._price_monte_carlo(n_paths, n_steps)

    def _price_monte_carlo(self, n_paths=50000, n_steps=252):
        """
        Pricing par Monte Carlo.

        Méthode:
            1. Simuler n_paths chemins de prix
            2. Pour chaque chemin, vérifier si la barrière est touchée
            3. Calculer le payoff selon le type (knock-in/out)
            4. Actualiser et moyenner

        Retourne:
        ---------
        float : prix Monte Carlo
        """
        # Simuler les chemins
        paths = simulate_gbm_paths(self.S, self.r, self.sigma, self.T,
                                   n_steps, n_paths)

        # Déterminer si la barrière est touchée pour chaque chemin
        if 'down' in self.barrier_type:
            barrier_touched = np.min(paths, axis=1) <= self.barrier
        else:  # up
            barrier_touched = np.max(paths, axis=1) >= self.barrier

        # Prix final (dernière colonne)
        final_prices = paths[:, -1]

        # Calculer les payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(final_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - final_prices, 0)

        # Appliquer la logique de barrière
        if 'out' in self.barrier_type:
            # Knock-out: payoff = 0 si barrière touchée
            payoffs = payoffs * (~barrier_touched)
        else:  # 'in'
            # Knock-in: payoff seulement si barrière touchée
            payoffs = payoffs * barrier_touched

        # Prix = espérance actualisée
        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)

        return price

    def _price_analytical(self):
        """
        Pricing analytique (formules fermées pour certains cas).

        Formules de Merton-Reiner-Rubinstein (Hull, Section 26.3)

        Note: Formules complexes, simplifié ici pour down-and-out call.
        """
        # Lambda et y pour les formules
        lambda_param = (self.r + self.sigma**2 / 2) / self.sigma**2
        y = np.log((self.barrier**2) / (self.S * self.K)) / \
            (self.sigma * np.sqrt(self.T)) + lambda_param * self.sigma * np.sqrt(self.T)

        x1 = np.log(self.S / self.barrier) / (self.sigma * np.sqrt(self.T)) + \
             lambda_param * self.sigma * np.sqrt(self.T)
        y1 = np.log(self.barrier / self.S) / (self.sigma * np.sqrt(self.T)) + \
             lambda_param * self.sigma * np.sqrt(self.T)

        # Pour down-and-out call avec K > H (barrière)
        if self.barrier_type == 'down-and-out' and self.option_type == 'call':
            if self.K > self.barrier:
                # Formule simplifiée (Hull, Table 26.2)
                d_1 = d1(self.S, self.K, self.T, self.r, self.sigma)
                d_2 = d2(self.S, self.K, self.T, self.r, self.sigma)

                vanilla = (self.S * norm.cdf(d_1) -
                          self.K * np.exp(-self.r * self.T) * norm.cdf(d_2))

                # Terme de rebate (approximation)
                rebate_term = (self.barrier / self.S) ** (2 * lambda_param) * \
                              self.S * norm.cdf(y) - \
                              self.K * np.exp(-self.r * self.T) * \
                              (self.barrier / self.S) ** (2 * lambda_param - 2) * \
                              norm.cdf(y - self.sigma * np.sqrt(self.T))

                return max(vanilla - rebate_term, 0)

        # Pour les autres cas, utiliser Monte Carlo
        return self._price_monte_carlo()

    def barrier_probability(self, n_paths=50000, n_steps=252):
        """
        Estime la probabilité que la barrière soit touchée.

        Retourne:
        ---------
        float : probabilité de toucher la barrière
        """
        paths = simulate_gbm_paths(self.S, self.r, self.sigma, self.T,
                                   n_steps, n_paths)

        if 'down' in self.barrier_type:
            barrier_touched = np.min(paths, axis=1) <= self.barrier
        else:
            barrier_touched = np.max(paths, axis=1) >= self.barrier

        return np.mean(barrier_touched)

    def description(self):
        """
        Description de l'option barrière.
        """
        return (f"Barrier {self.option_type.upper()} ({self.barrier_type}): "
                f"S={self.S}, K={self.K}, B={self.barrier}, "
                f"T={self.T:.2f}ans, σ={self.sigma:.1%}")


def in_out_parity_check(S, K, T, r, sigma, barrier, option_type):
    """
    Vérifie la parité In-Out.

    Propriété (Hull, Section 26.3):
        Prix(Knock-In) + Prix(Knock-Out) = Prix(Vanille)

    Paramètres:
    -----------
    S, K, T, r, sigma : paramètres standard
    barrier : float
        Niveau de barrière
    option_type : str
        'call' ou 'put'

    Retourne:
    ---------
    dict : résultat de la vérification
    """
    from .vanilla_options import EuropeanOption

    # Prix vanille
    vanilla = EuropeanOption(S, K, T, r, sigma, option_type)
    vanilla_price = vanilla.price()

    # Déterminer si down ou up
    if barrier < S:
        barrier_in = BarrierOption(S, K, T, r, sigma, barrier, 'down-and-in', option_type)
        barrier_out = BarrierOption(S, K, T, r, sigma, barrier, 'down-and-out', option_type)
    else:
        barrier_in = BarrierOption(S, K, T, r, sigma, barrier, 'up-and-in', option_type)
        barrier_out = BarrierOption(S, K, T, r, sigma, barrier, 'up-and-out', option_type)

    in_price = barrier_in.price()
    out_price = barrier_out.price()

    sum_price = in_price + out_price
    difference = abs(sum_price - vanilla_price)

    return {
        'vanilla_price': vanilla_price,
        'knock_in_price': in_price,
        'knock_out_price': out_price,
        'sum_in_out': sum_price,
        'difference': difference,
        'parity_holds': difference / vanilla_price < 0.05  # Tolérance 5%
    }


# Tests si exécuté directement
if __name__ == "__main__":
    print("=== Test Options Barrières ===")

    # Paramètres de test
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.25

    # Down-and-Out Call (barrière à 80)
    barrier_down = 80
    dao_call = BarrierOption(S, K, T, r, sigma, barrier_down, 'down-and-out', 'call')
    dai_call = BarrierOption(S, K, T, r, sigma, barrier_down, 'down-and-in', 'call')

    print(f"\n--- Barrière Down (B={barrier_down}) ---")
    print(f"Down-and-Out Call: {dao_call.price():.4f} €")
    print(f"Down-and-In Call:  {dai_call.price():.4f} €")
    print(f"P(toucher barrière): {dao_call.barrier_probability():.2%}")

    # Up-and-Out Call (barrière à 120)
    barrier_up = 120
    uao_call = BarrierOption(S, K, T, r, sigma, barrier_up, 'up-and-out', 'call')
    uai_call = BarrierOption(S, K, T, r, sigma, barrier_up, 'up-and-in', 'call')

    print(f"\n--- Barrière Up (B={barrier_up}) ---")
    print(f"Up-and-Out Call: {uao_call.price():.4f} €")
    print(f"Up-and-In Call:  {uai_call.price():.4f} €")

    # Vérification parité In-Out
    print(f"\n--- Parité In-Out (Down, B={barrier_down}) ---")
    parity = in_out_parity_check(S, K, T, r, sigma, barrier_down, 'call')
    print(f"Vanille: {parity['vanilla_price']:.4f}")
    print(f"In + Out: {parity['sum_in_out']:.4f}")
    print(f"Parité respectée: {parity['parity_holds']}")

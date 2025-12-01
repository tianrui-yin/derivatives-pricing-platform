# -*- coding: utf-8 -*-
"""
Options Vanilles
================
Options européennes et américaines.

Référence: Hull, Chapitres 13 (arbres binomiaux) et 15 (Black-Scholes)

Options européennes: exercice uniquement à maturité
Options américaines: exercice à tout moment
"""

import numpy as np
from scipy.stats import norm
from .base_product import BaseOption
from .utils import black_scholes_price, d1, d2


class EuropeanOption(BaseOption):
    """
    Option européenne (exercice uniquement à maturité).

    Pricing: Formule de Black-Scholes-Merton.

    Exemple:
    --------
    >>> call = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
    >>> print(f"Prix: {call.price():.4f}")
    """

    def __init__(self, S, K, T, r, sigma, option_type='call'):
        """
        Initialise une option européenne.

        Paramètres:
        -----------
        S : float
            Prix spot du sous-jacent
        K : float
            Prix d'exercice (strike)
        T : float
            Temps jusqu'à maturité (années)
        r : float
            Taux sans risque
        sigma : float
            Volatilité
        option_type : str
            'call' ou 'put'
        """
        super().__init__(S, K, T, r, sigma, option_type, name="European Option")

    def price(self):
        """
        Calcule le prix de l'option par Black-Scholes.

        Formules (Hull, Eq. 15.20-15.21):
            Call: c = S*N(d1) - K*e^(-rT)*N(d2)
            Put:  p = K*e^(-rT)*N(-d2) - S*N(-d1)

        Retourne:
        ---------
        float : prix de l'option
        """
        return black_scholes_price(self.S, self.K, self.T, self.r, self.sigma,
                                   self.option_type)

    def delta(self):
        """
        Calcule le Delta de l'option.

        Retourne:
        ---------
        float : Delta
        """
        d_1 = d1(self.S, self.K, self.T, self.r, self.sigma)
        if self.option_type == 'call':
            return norm.cdf(d_1)
        else:
            return norm.cdf(d_1) - 1

    def gamma(self):
        """
        Calcule le Gamma de l'option.

        Retourne:
        ---------
        float : Gamma
        """
        d_1 = d1(self.S, self.K, self.T, self.r, self.sigma)
        return norm.pdf(d_1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        """
        Calcule le Vega de l'option (pour 1% de vol).

        Retourne:
        ---------
        float : Vega
        """
        d_1 = d1(self.S, self.K, self.T, self.r, self.sigma)
        return self.S * np.sqrt(self.T) * norm.pdf(d_1) / 100


class AmericanOption(BaseOption):
    """
    Option américaine (exercice à tout moment).

    Pricing: Arbre binomial (méthode CRR - Cox-Ross-Rubinstein).

    Référence: Hull, Chapitre 13

    L'exercice anticipé est optimal pour:
        - Put américain sur action sans dividende (parfois)
        - Options sur actions avec dividendes

    Exemple:
    --------
    >>> put = AmericanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
    >>> print(f"Prix: {put.price():.4f}")
    """

    def __init__(self, S, K, T, r, sigma, option_type='call', n_steps=100):
        """
        Initialise une option américaine.

        Paramètres:
        -----------
        S, K, T, r, sigma, option_type : paramètres standard
        n_steps : int
            Nombre de pas dans l'arbre binomial (défaut: 100)
        """
        super().__init__(S, K, T, r, sigma, option_type, name="American Option")
        self.n_steps = n_steps

    def price(self):
        """
        Calcule le prix par arbre binomial CRR.

        Méthode (Hull, Section 13.1):
            1. Construire l'arbre des prix du sous-jacent
            2. Calculer les payoffs à maturité
            3. Remonter l'arbre en comparant:
               - Valeur d'attente (actualisation du noeud suivant)
               - Valeur d'exercice (payoff immédiat)

        Paramètres CRR:
            u = e^(σ√dt)  : mouvement haussier
            d = 1/u       : mouvement baissier
            p = (e^(rdt) - d) / (u - d) : probabilité risque-neutre

        Retourne:
        ---------
        float : prix de l'option
        """
        dt = self.T / self.n_steps

        # Paramètres CRR
        u = np.exp(self.sigma * np.sqrt(dt))  # Mouvement up
        d = 1 / u                              # Mouvement down
        p = (np.exp(self.r * dt) - d) / (u - d)  # Probabilité risque-neutre

        # Facteur d'actualisation
        discount = np.exp(-self.r * dt)

        # Initialiser les prix du sous-jacent à maturité
        # À l'étape n, il y a n+1 noeuds possibles
        stock_prices = np.zeros(self.n_steps + 1)
        for j in range(self.n_steps + 1):
            # Nombre de mouvements up = j, down = n-j
            stock_prices[j] = self.S * (u ** j) * (d ** (self.n_steps - j))

        # Calculer les payoffs à maturité
        if self.option_type == 'call':
            option_values = np.maximum(stock_prices - self.K, 0)
        else:
            option_values = np.maximum(self.K - stock_prices, 0)

        # Remonter l'arbre (backward induction)
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Prix du sous-jacent à ce noeud
                S_node = self.S * (u ** j) * (d ** (i - j))

                # Valeur d'exercice immédiat
                if self.option_type == 'call':
                    exercise_value = max(S_node - self.K, 0)
                else:
                    exercise_value = max(self.K - S_node, 0)

                # Valeur de continuation (actualisation)
                continuation_value = discount * (p * option_values[j + 1] +
                                                  (1 - p) * option_values[j])

                # Pour option américaine: max(exercice, continuation)
                option_values[j] = max(exercise_value, continuation_value)

        return option_values[0]

    def early_exercise_premium(self):
        """
        Calcule la prime d'exercice anticipé.

        C'est la différence entre le prix américain et européen.

        Retourne:
        ---------
        float : prime d'exercice anticipé
        """
        american_price = self.price()
        european_price = black_scholes_price(self.S, self.K, self.T, self.r,
                                              self.sigma, self.option_type)
        return american_price - european_price


def binomial_tree_visualization(S, K, T, r, sigma, option_type, n_steps=4):
    """
    Génère une visualisation textuelle de l'arbre binomial.

    Utile pour comprendre la méthode et pour les entretiens.

    Paramètres:
    -----------
    S, K, T, r, sigma, option_type : paramètres standard
    n_steps : int
        Nombre de pas (petit pour la visualisation)

    Retourne:
    ---------
    str : représentation textuelle de l'arbre
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    lines = [
        f"Arbre Binomial - {option_type.upper()} {n_steps} pas",
        f"Paramètres: S={S}, K={K}, T={T}, r={r}, σ={sigma}",
        f"u={u:.4f}, d={d:.4f}, p={p:.4f}",
        ""
    ]

    # Afficher les prix du sous-jacent
    lines.append("Prix du sous-jacent:")
    for i in range(n_steps + 1):
        level_prices = []
        for j in range(i + 1):
            price = S * (u ** j) * (d ** (i - j))
            level_prices.append(f"{price:.2f}")
        lines.append(f"  t={i}: " + " | ".join(level_prices))

    return '\n'.join(lines)


# Tests si exécuté directement
if __name__ == "__main__":
    print("=== Test Options Vanilles ===")

    # Paramètres de test
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.20

    # Option européenne
    euro_call = EuropeanOption(S, K, T, r, sigma, 'call')
    euro_put = EuropeanOption(S, K, T, r, sigma, 'put')

    print(f"\n--- Options Européennes ---")
    print(f"Call: {euro_call.price():.4f} €")
    print(f"Put:  {euro_put.price():.4f} €")
    print(f"Delta Call: {euro_call.delta():.4f}")
    print(f"Gamma: {euro_call.gamma():.6f}")
    print(f"Vega: {euro_call.vega():.4f}")

    # Option américaine
    amer_call = AmericanOption(S, K, T, r, sigma, 'call', n_steps=200)
    amer_put = AmericanOption(S, K, T, r, sigma, 'put', n_steps=200)

    print(f"\n--- Options Américaines ---")
    print(f"Call: {amer_call.price():.4f} €")
    print(f"Put:  {amer_put.price():.4f} €")
    print(f"Prime exercice anticipé (put): {amer_put.early_exercise_premium():.4f} €")

    # Vérification: Call américain = Call européen (sans dividende)
    diff_call = abs(amer_call.price() - euro_call.price())
    print(f"\nDifférence Call US - EU: {diff_call:.6f} (devrait être ~0)")

    # Visualisation
    print(f"\n{binomial_tree_visualization(S, K, T, r, sigma, 'call', n_steps=3)}")

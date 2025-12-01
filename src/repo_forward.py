# -*- coding: utf-8 -*-
"""
Repos et Forwards
=================
Accords de rachat (Repo) et contrats à terme (Forwards).

Référence: Hull, Chapitres 4 et 5

Repo (Repurchase Agreement):
    - Vente d'un titre avec engagement de rachat à une date future
    - Équivalent économique d'un prêt collatéralisé

Forward:
    - Contrat d'achat/vente à une date future à un prix fixé aujourd'hui
    - Pas de flux initial (contrairement aux futures)
"""

import numpy as np
from .base_product import BaseDerivative


class Repo(BaseDerivative):
    """
    Accord de rachat (Repurchase Agreement).

    Un Repo est économiquement équivalent à un prêt collatéralisé:
        - Le vendeur reçoit du cash et donne des titres en garantie
        - À maturité, il rachète les titres et rembourse le cash + intérêts

    Le taux repo est le coût de financement implicite.

    Référence: Hull, Section 4.1

    Exemple:
    --------
    >>> repo = Repo(principal=10000000, repo_rate=0.02, maturity=7/365)
    >>> print(f"Montant de rachat: {repo.repurchase_price():,.2f} €")
    """

    def __init__(self, principal, repo_rate, maturity, collateral_value=None,
                 haircut=0.02):
        """
        Initialise un Repo.

        Paramètres:
        -----------
        principal : float
            Montant du cash échangé (prix de vente initial)
        repo_rate : float
            Taux repo (annualisé)
        maturity : float
            Maturité en années (ex: 7/365 pour 7 jours)
        collateral_value : float
            Valeur de marché du collatéral (optionnel)
        haircut : float
            Décote appliquée au collatéral (ex: 0.02 = 2%)
        """
        super().__init__(name="Repo Agreement", notional=principal)
        self.principal = principal
        self.repo_rate = repo_rate
        self.maturity = maturity
        self.haircut = haircut

        # Valeur du collatéral (par défaut: principal / (1 - haircut))
        if collateral_value is None:
            self.collateral_value = principal / (1 - haircut)
        else:
            self.collateral_value = collateral_value

    def repurchase_price(self):
        """
        Calcule le prix de rachat à maturité.

        Formule:
            Prix de rachat = Principal * (1 + repo_rate * T)

        Retourne:
        ---------
        float : montant à payer pour récupérer le collatéral
        """
        return self.principal * (1 + self.repo_rate * self.maturity)

    def interest_amount(self):
        """
        Calcule les intérêts du repo.

        Retourne:
        ---------
        float : montant des intérêts
        """
        return self.principal * self.repo_rate * self.maturity

    def implied_repo_rate(self, spot_price, forward_price, T):
        """
        Calcule le taux repo implicite à partir des prix spot et forward.

        Formule (Hull, Eq. 5.1):
            r_repo = (F/S - 1) / T

        Paramètres:
        -----------
        spot_price : float
            Prix spot du sous-jacent
        forward_price : float
            Prix forward
        T : float
            Maturité du forward

        Retourne:
        ---------
        float : taux repo implicite
        """
        return (forward_price / spot_price - 1) / T

    def price(self):
        """
        Calcule la valeur actuelle du repo (pour un taux de marché donné).

        La valeur dépend de la différence entre le taux contractuel
        et le taux de marché actuel.

        Retourne:
        ---------
        float : valeur du repo
        """
        # Pour un repo à son initiation, la valeur est 0
        # La valeur change si les taux de marché changent
        return 0  # Simplification

    def description(self):
        """
        Description du Repo.
        """
        days = self.maturity * 365
        return (f"Repo: Principal={self.principal:,.0f}€, "
                f"Taux={self.repo_rate:.2%}, "
                f"Durée={days:.0f}j, "
                f"Rachat={self.repurchase_price():,.2f}€")


class Forward(BaseDerivative):
    """
    Contrat Forward.

    Un Forward est un engagement d'acheter (long) ou vendre (short)
    un actif à une date future à un prix fixé aujourd'hui.

    Caractéristiques:
        - Pas de flux initial (contrairement aux options)
        - Contrat OTC (non standardisé)
        - Risque de contrepartie

    Référence: Hull, Chapitre 5

    Exemple:
    --------
    >>> forward = Forward(S=100, r=0.05, T=1, q=0.02)
    >>> print(f"Prix Forward: {forward.forward_price():.4f} €")
    """

    def __init__(self, S, r, T, q=0, delivery_price=None):
        """
        Initialise un Forward.

        Paramètres:
        -----------
        S : float
            Prix spot actuel du sous-jacent
        r : float
            Taux sans risque (annualisé)
        T : float
            Temps jusqu'à maturité (années)
        q : float
            Taux de dividende continu (défaut: 0)
        delivery_price : float
            Prix de livraison (K). Si None, calculé comme F (prix juste)
        """
        super().__init__(name="Forward Contract")
        self.S = S
        self.r = r
        self.T = T
        self.q = q

        # Si pas de prix de livraison spécifié, utiliser le prix forward juste
        if delivery_price is None:
            self.K = self.forward_price()
        else:
            self.K = delivery_price

    def forward_price(self):
        """
        Calcule le prix forward théorique.

        Formule (Hull, Eq. 5.3):
            F = S * e^((r-q)T)

        C'est le prix de livraison qui rend la valeur du contrat nulle.

        Retourne:
        ---------
        float : prix forward
        """
        return self.S * np.exp((self.r - self.q) * self.T)

    def price(self, position='long'):
        """
        Calcule la valeur du contrat forward.

        Formule (Hull, Eq. 5.5):
            f_long = S*e^(-qT) - K*e^(-rT)
            f_short = K*e^(-rT) - S*e^(-qT)

        Paramètres:
        -----------
        position : str
            'long' (acheteur) ou 'short' (vendeur)

        Retourne:
        ---------
        float : valeur du contrat
        """
        pv_spot = self.S * np.exp(-self.q * self.T)
        pv_strike = self.K * np.exp(-self.r * self.T)

        if position.lower() == 'long':
            return pv_spot - pv_strike
        else:
            return pv_strike - pv_spot

    def payoff(self, S_T, position='long'):
        """
        Calcule le payoff à maturité.

        Payoff long = S_T - K
        Payoff short = K - S_T

        Paramètres:
        -----------
        S_T : float
            Prix spot à maturité
        position : str
            'long' ou 'short'

        Retourne:
        ---------
        float : payoff
        """
        if position.lower() == 'long':
            return S_T - self.K
        else:
            return self.K - S_T

    def break_even_spot(self):
        """
        Calcule le prix spot à maturité pour break-even (payoff = 0).

        Pour un forward, c'est simplement le prix de livraison K.

        Retourne:
        ---------
        float : prix break-even
        """
        return self.K

    def delta(self):
        """
        Calcule le Delta du forward.

        Pour un forward long, Delta = e^(-qT) ≈ 1

        Retourne:
        ---------
        float : Delta
        """
        return np.exp(-self.q * self.T)

    def description(self):
        """
        Description du Forward.
        """
        return (f"Forward: S={self.S:.2f}, K={self.K:.2f}, "
                f"T={self.T:.2f}ans, r={self.r:.2%}, q={self.q:.2%}")


def cost_of_carry(r, q, storage_cost=0, convenience_yield=0):
    """
    Calcule le coût de portage (cost of carry).

    Le coût de portage est la différence entre le prix forward et spot:
        F = S * e^(c*T)

    où c = r - q + u - y (Hull, Section 5.10)

    Paramètres:
    -----------
    r : float
        Taux sans risque
    q : float
        Rendement du dividende
    storage_cost : float
        Coût de stockage (pour commodités)
    convenience_yield : float
        Rendement de commodité (pour commodités)

    Retourne:
    ---------
    float : coût de portage
    """
    return r - q + storage_cost - convenience_yield


def forward_vs_futures_spread(forward_price, futures_price):
    """
    Compare les prix forward et futures.

    En théorie, si les taux sont déterministes:
        F_forward = F_futures

    Mais en pratique, des différences peuvent exister
    (convexity bias, liquidité, etc.)

    Paramètres:
    -----------
    forward_price : float
        Prix forward
    futures_price : float
        Prix futures

    Retourne:
    ---------
    dict : spread et ratio
    """
    spread = forward_price - futures_price
    ratio = forward_price / futures_price if futures_price != 0 else np.nan

    return {
        'forward': forward_price,
        'futures': futures_price,
        'spread': spread,
        'ratio': ratio
    }


# Tests si exécuté directement
if __name__ == "__main__":
    print("=== Test Repos et Forwards ===")

    # Repo
    print(f"\n--- Repo ---")
    principal = 10_000_000  # 10 millions €
    repo_rate = 0.015  # 1.5% annualisé
    maturity_days = 7

    repo = Repo(principal, repo_rate, maturity_days / 365)
    print(repo.description())
    print(f"Intérêts: {repo.interest_amount():,.2f} €")
    print(f"Prix de rachat: {repo.repurchase_price():,.2f} €")

    # Forward sur action
    print(f"\n--- Forward sur Action ---")
    S = 100  # Prix spot
    r = 0.05  # Taux sans risque 5%
    T = 1.0  # 1 an
    q = 0.02  # Dividende 2%

    forward = Forward(S, r, T, q)
    print(f"Prix spot: {S:.2f} €")
    print(f"Prix forward théorique: {forward.forward_price():.4f} €")
    print(f"Valeur du contrat long: {forward.price('long'):.4f} € (devrait être ~0)")

    # Forward avec prix de livraison différent
    print(f"\n--- Forward avec K ≠ F ---")
    K = 105  # Prix de livraison différent du forward juste
    forward_k = Forward(S, r, T, q, delivery_price=K)
    print(f"Prix de livraison: {K:.2f} €")
    print(f"Prix forward théorique: {forward_k.forward_price():.4f} €")
    print(f"Valeur du contrat long: {forward_k.price('long'):.4f} €")
    print(f"Valeur du contrat short: {forward_k.price('short'):.4f} €")

    # Payoff à différents prix finaux
    print(f"\n--- Payoffs à Maturité ---")
    for S_T in [90, 100, 110, 120]:
        payoff_long = forward.payoff(S_T, 'long')
        print(f"S_T = {S_T}: Payoff long = {payoff_long:+.2f} €")

    # Coût de portage
    print(f"\n--- Coût de Portage ---")
    c = cost_of_carry(r, q)
    print(f"Cost of carry: {c:.2%}")
    print(f"Vérification: S*e^(cT) = {S * np.exp(c * T):.4f} vs F = {forward.forward_price():.4f}")

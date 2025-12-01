# -*- coding: utf-8 -*-
"""
Classe de Base pour les Produits Dérivés
========================================
Architecture OOP extensible pour tous les produits.

Cette classe abstraite définit l'interface commune à tous les
produits dérivés de la plateforme.
"""

from abc import ABC, abstractmethod


class BaseDerivative(ABC):
    """
    Classe abstraite de base pour tous les produits dérivés.

    Attributs communs:
    ------------------
    name : str
        Nom du produit
    currency : str
        Devise (EUR par défaut)
    notional : float
        Nominal du contrat

    Méthodes abstraites:
    --------------------
    price() : Calcule le prix du produit
    description() : Retourne une description du produit
    """

    def __init__(self, name="Derivative", currency="EUR", notional=1.0):
        """
        Initialise le produit dérivé.

        Paramètres:
        -----------
        name : str
            Nom du produit
        currency : str
            Devise
        notional : float
            Nominal du contrat
        """
        self.name = name
        self.currency = currency
        self.notional = notional

    @abstractmethod
    def price(self):
        """
        Calcule le prix du produit dérivé.

        Cette méthode doit être implémentée par chaque produit.

        Retourne:
        ---------
        float : prix du produit
        """
        pass

    @abstractmethod
    def description(self):
        """
        Retourne une description textuelle du produit.

        Retourne:
        ---------
        str : description du produit
        """
        pass

    def __str__(self):
        """
        Représentation textuelle du produit.
        """
        return f"{self.name} ({self.currency})"

    def __repr__(self):
        """
        Représentation pour le débogage.
        """
        return f"{self.__class__.__name__}(name='{self.name}')"


class BaseOption(BaseDerivative):
    """
    Classe de base pour toutes les options.

    Attributs:
    ----------
    S : float
        Prix spot du sous-jacent
    K : float
        Prix d'exercice (strike)
    T : float
        Temps jusqu'à maturité (en années)
    r : float
        Taux sans risque
    sigma : float
        Volatilité
    option_type : str
        'call' ou 'put'
    """

    def __init__(self, S, K, T, r, sigma, option_type='call', **kwargs):
        """
        Initialise une option.

        Paramètres:
        -----------
        S : float
            Prix spot
        K : float
            Strike
        T : float
            Maturité (années)
        r : float
            Taux sans risque
        sigma : float
            Volatilité
        option_type : str
            'call' ou 'put'
        """
        super().__init__(**kwargs)
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()

        # Validation
        self._validate_params()

    def _validate_params(self):
        """
        Valide les paramètres de l'option.
        """
        if self.S <= 0:
            raise ValueError("Le prix spot (S) doit être positif")
        if self.K <= 0:
            raise ValueError("Le strike (K) doit être positif")
        if self.T < 0:
            raise ValueError("La maturité (T) doit être positive ou nulle")
        if self.sigma < 0:
            raise ValueError("La volatilité (sigma) doit être positive")
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type doit être 'call' ou 'put'")

    def intrinsic_value(self):
        """
        Calcule la valeur intrinsèque de l'option.

        Retourne:
        ---------
        float : max(S-K, 0) pour call, max(K-S, 0) pour put
        """
        if self.option_type == 'call':
            return max(self.S - self.K, 0)
        else:
            return max(self.K - self.S, 0)

    def is_in_the_money(self):
        """
        Détermine si l'option est dans la monnaie (ITM).

        Retourne:
        ---------
        bool : True si ITM
        """
        return self.intrinsic_value() > 0

    def moneyness(self):
        """
        Calcule le moneyness de l'option.

        Retourne:
        ---------
        float : K/S
        """
        return self.K / self.S

    def description(self):
        """
        Description de l'option.
        """
        status = "ITM" if self.is_in_the_money() else "OTM"
        return (f"{self.option_type.upper()} Option: "
                f"S={self.S}, K={self.K}, T={self.T:.2f}ans, "
                f"σ={self.sigma:.1%}, {status}")


class BaseInterestRateProduct(BaseDerivative):
    """
    Classe de base pour les produits de taux.

    Attributs:
    ----------
    notional : float
        Nominal du contrat
    fixed_rate : float
        Taux fixe
    start_date : float
        Date de début (en années depuis aujourd'hui)
    end_date : float
        Date de fin
    """

    def __init__(self, notional, fixed_rate, start_date, end_date, **kwargs):
        """
        Initialise un produit de taux.

        Paramètres:
        -----------
        notional : float
            Nominal
        fixed_rate : float
            Taux fixe du contrat
        start_date : float
            Date de début
        end_date : float
            Date de fin
        """
        super().__init__(notional=notional, **kwargs)
        self.fixed_rate = fixed_rate
        self.start_date = start_date
        self.end_date = end_date

    def description(self):
        """
        Description du produit de taux.
        """
        return (f"Produit de taux: Nominal={self.notional:,.0f}, "
                f"Taux fixe={self.fixed_rate:.2%}, "
                f"Durée={self.end_date - self.start_date:.1f} ans")

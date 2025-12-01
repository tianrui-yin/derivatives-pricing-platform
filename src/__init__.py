# -*- coding: utf-8 -*-
"""
Derivatives Pricing Platform
============================
Plateforme multi-produits de pricing de dérivés.

Modules:
    - base_product: Classe de base abstraite
    - vanilla_options: Options européennes et américaines
    - barrier_options: Options barrières
    - asian_options: Options asiatiques
    - lookback_options: Options lookback
    - interest_rate: IRS, Caps, Floors
    - repo_forward: Repos et Forwards
    - utils: Fonctions utilitaires
"""

from .vanilla_options import EuropeanOption, AmericanOption
from .barrier_options import BarrierOption
from .asian_options import AsianOption
from .lookback_options import LookbackOption
from .interest_rate import InterestRateSwap, Cap, Floor
from .repo_forward import Repo, Forward

__version__ = "1.0.0"
__author__ = "M2 Finance Quantitative - Sorbonne"

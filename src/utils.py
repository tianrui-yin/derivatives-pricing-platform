# -*- coding: utf-8 -*-
"""
Fonctions Utilitaires
=====================
Fonctions communes utilisées par plusieurs modules.
"""

import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    """
    Calcule d1 dans la formule Black-Scholes.

    Formule (Hull, Eq. 15.20):
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)

    Paramètres:
    -----------
    S, K, T, r, sigma : paramètres BSM standard

    Retourne:
    ---------
    float : valeur de d1
    """
    if T <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    """
    Calcule d2 dans la formule Black-Scholes.

    Formule: d2 = d1 - σ√T

    Paramètres:
    -----------
    S, K, T, r, sigma : paramètres BSM standard

    Retourne:
    ---------
    float : valeur de d2
    """
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calcule le prix Black-Scholes d'une option européenne.

    Formules (Hull, Eq. 15.20-15.21):
        Call: c = S*N(d1) - K*e^(-rT)*N(d2)
        Put:  p = K*e^(-rT)*N(-d2) - S*N(-d1)

    Paramètres:
    -----------
    S, K, T, r, sigma : paramètres standard
    option_type : 'call' ou 'put'

    Retourne:
    ---------
    float : prix de l'option
    """
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    discount = np.exp(-r * T)

    if option_type.lower() == 'call':
        return S * norm.cdf(d_1) - K * discount * norm.cdf(d_2)
    else:
        return K * discount * norm.cdf(-d_2) - S * norm.cdf(-d_1)


def simulate_gbm_path(S0, r, sigma, T, n_steps, seed=None):
    """
    Simule un chemin de prix (Mouvement Brownien Géométrique).

    Modèle (Hull, Eq. 14.14):
        S(t+dt) = S(t) * exp((r - σ²/2)*dt + σ*√dt*Z)

    Paramètres:
    -----------
    S0 : float
        Prix initial
    r : float
        Drift (taux sans risque)
    sigma : float
        Volatilité
    T : float
        Horizon
    n_steps : int
        Nombre de pas
    seed : int
        Graine aléatoire

    Retourne:
    ---------
    np.array : chemin de prix
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    path = np.zeros(n_steps + 1)
    path[0] = S0

    for i in range(1, n_steps + 1):
        Z = np.random.standard_normal()
        path[i] = path[i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return path


def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=None):
    """
    Simule plusieurs chemins de prix (Monte Carlo).

    Paramètres:
    -----------
    S0, r, sigma, T, n_steps : voir simulate_gbm_path
    n_paths : int
        Nombre de simulations
    seed : int
        Graine aléatoire

    Retourne:
    ---------
    np.array : matrice de chemins (n_paths x n_steps+1)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for i in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        paths[:, i] = paths[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return paths


def discount_factor(r, T):
    """
    Calcule le facteur d'actualisation.

    Formule: DF = e^(-rT)

    Paramètres:
    -----------
    r : float
        Taux sans risque
    T : float
        Maturité

    Retourne:
    ---------
    float : facteur d'actualisation
    """
    return np.exp(-r * T)


def forward_price(S, r, T, q=0):
    """
    Calcule le prix forward.

    Formule (Hull, Eq. 5.3):
        F = S * e^((r-q)T)

    Paramètres:
    -----------
    S : float
        Prix spot
    r : float
        Taux sans risque
    T : float
        Maturité
    q : float
        Taux de dividende continu

    Retourne:
    ---------
    float : prix forward
    """
    return S * np.exp((r - q) * T)


def black_price(F, K, T, r, sigma, option_type='call'):
    """
    Modèle de Black pour les options sur forwards/futures.

    Utilisé pour les caps/floors.

    Formule (Hull, Eq. 29.1-29.2):
        Call: c = e^(-rT) * [F*N(d1) - K*N(d2)]
        Put:  p = e^(-rT) * [K*N(-d2) - F*N(-d1)]

    Paramètres:
    -----------
    F : float
        Prix forward
    K : float
        Strike
    T : float
        Maturité
    r : float
        Taux sans risque
    sigma : float
        Volatilité
    option_type : str
        'call' ou 'put'

    Retourne:
    ---------
    float : prix de l'option
    """
    if T <= 0:
        if option_type == 'call':
            return max(F - K, 0) * np.exp(-r * T)
        else:
            return max(K - F, 0) * np.exp(-r * T)

    d_1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)

    discount = np.exp(-r * T)

    if option_type.lower() == 'call':
        return discount * (F * norm.cdf(d_1) - K * norm.cdf(d_2))
    else:
        return discount * (K * norm.cdf(-d_2) - F * norm.cdf(-d_1))

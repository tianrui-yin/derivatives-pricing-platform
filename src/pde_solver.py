# -*- coding: utf-8 -*-
"""
Solveur PDE Crank-Nicolson pour Black-Scholes-Merton
=====================================================
Résolution de l'EDP de Black-Scholes par différences finies (Crank-Nicolson).

EDP de Black-Scholes-Merton:
    ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

Transformation en log-spot x = ln(S) pour grille uniforme:
    ∂V/∂t + ½σ²∂²V/∂x² + (r - ½σ²)∂V/∂x - rV = 0

Schéma Crank-Nicolson (θ = 0.5, moyenne implicite/explicite):
    [I - 0.5*dt*A] * V^{n} = [I + 0.5*dt*A] * V^{n+1}

Référence: Hull, Chapitre 21 (Finite Difference Methods)
"""

import numpy as np
import logging
from typing import Optional

from .utils import black_scholes_price, simulate_gbm_paths
from .vanilla_options import AmericanOption

logger = logging.getLogger(__name__)


class PDESolver:
    """
    Solveur Crank-Nicolson pour l'EDP de Black-Scholes-Merton.

    Résout l'EDP en remontant le temps de T à 0 sur une grille log-spot.

    Paramètres:
    -----------
    S0 : float
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
    exercise : str
        'european' ou 'american'
    N_space : int
        Nombre de points spatiaux (défaut 200)
    N_time : int
        Nombre de pas temporels (défaut 200)
    S_max_mult : float
        Borne supérieure = S_max_mult * S0 (défaut 4.0)
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        exercise: str = 'european',
        N_space: int = 200,
        N_time: int = 200,
        S_max_mult: float = 4.0,
    ):
        if S0 <= 0:
            raise ValueError("S0 doit être positif")
        if K <= 0:
            raise ValueError("K doit être positif")
        if T <= 0:
            raise ValueError("T doit être positif")
        if sigma <= 0:
            raise ValueError("sigma doit être positif")
        if option_type.lower() not in ('call', 'put'):
            raise ValueError("option_type doit être 'call' ou 'put'")
        if exercise.lower() not in ('european', 'american'):
            raise ValueError("exercise doit être 'european' ou 'american'")

        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.option_type = option_type.lower()
        self.exercise = exercise.lower()
        self.N_space = int(N_space)
        self.N_time = int(N_time)
        self.S_max_mult = float(S_max_mult)

    # ------------------------------------------------------------------
    # Grille
    # ------------------------------------------------------------------
    def _setup_grid(self):
        """
        Crée la grille log-spot et la grille temporelle.

        x = ln(S), uniformément espacé de ln(S_min) à ln(S_max).
        S_min = S0 / S_max_mult, S_max = S0 * S_max_mult.
        """
        self.S_min = self.S0 / self.S_max_mult
        self.S_max = self.S0 * self.S_max_mult

        self.x_min = np.log(self.S_min)
        self.x_max = np.log(self.S_max)

        # Grille spatiale (N_space + 1 points, incluant les bords)
        self.x = np.linspace(self.x_min, self.x_max, self.N_space + 1)
        self.dx = self.x[1] - self.x[0]

        # Prix spot correspondants
        self.S_grid = np.exp(self.x)

        # Grille temporelle
        self.dt = self.T / self.N_time
        self.t_grid = np.linspace(0, self.T, self.N_time + 1)

    # ------------------------------------------------------------------
    # Conditions aux bords
    # ------------------------------------------------------------------
    def _boundary_conditions(self, t: float):
        """
        Conditions aux bords à S_min et S_max.

        Paramètres:
        -----------
        t : float
            Temps courant (0 = début, T = maturité)

        Retourne:
        ---------
        tuple (V_min, V_max) : valeurs aux bords

        Pour un call:
            V(S_min, t) = 0
            V(S_max, t) = S_max - K * exp(-r * (T - t))
        Pour un put:
            V(S_min, t) = K * exp(-r * (T - t)) - S_min
            V(S_max, t) = 0
        """
        tau = self.T - t  # temps restant jusqu'à maturité
        discount = np.exp(-self.r * tau)

        if self.option_type == 'call':
            V_min = 0.0
            V_max = self.S_max - self.K * discount
        else:
            V_min = self.K * discount - self.S_min
            V_max = 0.0

        return V_min, V_max

    # ------------------------------------------------------------------
    # Condition terminale
    # ------------------------------------------------------------------
    def _terminal_condition(self):
        """
        Payoff terminal à t = T.

        Retourne:
        ---------
        np.ndarray : vecteur de payoffs sur la grille S
        """
        if self.option_type == 'call':
            return np.maximum(self.S_grid - self.K, 0.0)
        else:
            return np.maximum(self.K - self.S_grid, 0.0)

    # ------------------------------------------------------------------
    # Matrice tridiagonale
    # ------------------------------------------------------------------
    def _build_tridiagonal(self):
        """
        Construit les diagonales de l'opérateur spatial A en coordonnées log-spot.

        Discrétisation par différences centrées:
            a_j = ½σ²/dx² - (r - ½σ²) / (2*dx)    (diag. inférieure)
            b_j = -σ²/dx² - r                        (diag. principale)
            c_j = ½σ²/dx² + (r - ½σ²) / (2*dx)     (diag. supérieure)

        Retourne:
        ---------
        tuple (a, b, c) : diagonales de taille N_interior
        """
        N_int = self.N_space - 1  # points intérieurs (sans les bords)
        dx = self.dx
        sig2 = self.sigma ** 2
        drift = self.r - 0.5 * sig2

        a = 0.5 * sig2 / dx**2 - drift / (2.0 * dx)
        b = -sig2 / dx**2 - self.r
        c = 0.5 * sig2 / dx**2 + drift / (2.0 * dx)

        lower = np.full(N_int, a)
        main = np.full(N_int, b)
        upper = np.full(N_int, c)

        return lower, main, upper

    # ------------------------------------------------------------------
    # Algorithme de Thomas (résolution tridiagonale O(n))
    # ------------------------------------------------------------------
    def _thomas_algorithm(
        self,
        lower: np.ndarray,
        main: np.ndarray,
        upper: np.ndarray,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """
        Algorithme de Thomas pour un système tridiagonal A*x = d.

        Complexité O(n).

        Paramètres:
        -----------
        lower : np.ndarray
            Diagonale inférieure (lower[0] n'est pas utilisé)
        main : np.ndarray
            Diagonale principale
        upper : np.ndarray
            Diagonale supérieure (upper[-1] n'est pas utilisé)
        rhs : np.ndarray
            Second membre

        Retourne:
        ---------
        np.ndarray : solution x
        """
        n = len(main)
        # Copies pour ne pas modifier les entrées
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)

        # Balayage avant (forward sweep)
        c_prime[0] = upper[0] / main[0]
        d_prime[0] = rhs[0] / main[0]

        for i in range(1, n):
            denom = main[i] - lower[i] * c_prime[i - 1]
            c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
            d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom

        # Balayage arrière (back substitution)
        x = np.zeros(n)
        x[-1] = d_prime[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    # ------------------------------------------------------------------
    # Résolution
    # ------------------------------------------------------------------
    def solve(self) -> float:
        """
        Résout l'EDP en remontant le temps de T à 0 par Crank-Nicolson.

        Pour les options américaines, à chaque pas de temps:
            V = max(V_continuation, valeur_intrinsèque)

        Retourne:
        ---------
        float : prix de l'option (interpolé à S0)
        """
        self._setup_grid()

        # Condition terminale (payoff à maturité)
        V = self._terminal_condition()

        # Construire les coefficients tridiagonaux
        lower, main, upper = self._build_tridiagonal()
        N_int = self.N_space - 1  # nombre de points intérieurs
        dt = self.dt

        # Matrices implicite et explicite (Crank-Nicolson, θ = 0.5)
        # Côté gauche : [I - 0.5*dt*A]
        lhs_lower = -0.5 * dt * lower
        lhs_main = 1.0 - 0.5 * dt * main
        lhs_upper = -0.5 * dt * upper

        # Côté droit : [I + 0.5*dt*A]
        rhs_lower = 0.5 * dt * lower
        rhs_main = 1.0 + 0.5 * dt * main
        rhs_upper = 0.5 * dt * upper

        # Valeur intrinsèque pour exercice américain
        if self.option_type == 'call':
            intrinsic = np.maximum(self.S_grid - self.K, 0.0)
        else:
            intrinsic = np.maximum(self.K - self.S_grid, 0.0)

        # Remonter le temps : de n = N_time (t=T) à n = 0 (t=0)
        for n in range(self.N_time - 1, -1, -1):
            t_n = self.t_grid[n]

            # Conditions aux bords au temps t_n
            bc_min, bc_max = self._boundary_conditions(t_n)

            # Points intérieurs : indices 1 à N_space-1
            V_int = V[1:self.N_space]  # taille N_int

            # Construire le second membre (côté explicite)
            rhs_vec = np.zeros(N_int)
            for j in range(N_int):
                rhs_vec[j] = rhs_main[j] * V_int[j]
                if j > 0:
                    rhs_vec[j] += rhs_lower[j] * V_int[j - 1]
                else:
                    rhs_vec[j] += rhs_lower[j] * V[0]  # bord gauche ancien
                if j < N_int - 1:
                    rhs_vec[j] += rhs_upper[j] * V_int[j + 1]
                else:
                    rhs_vec[j] += rhs_upper[j] * V[self.N_space]  # bord droit ancien

            # Incorporer les conditions aux bords dans le second membre
            # (contribution du côté implicite)
            rhs_vec[0] -= lhs_lower[0] * bc_min
            rhs_vec[-1] -= lhs_upper[-1] * bc_max

            # Incorporer les conditions aux bords dans le second membre
            # (contribution du côté explicite — bords actualisés)
            # On remplace les termes de bord explicites par les bords au temps n+1
            # puis on corrige avec les bords au temps n
            # Simplifié : on a déjà utilisé V[0] et V[N_space] ci-dessus (bords au temps n+1)
            # Maintenant on met à jour les bords et on ajuste

            # Résoudre le système tridiagonal
            V_new_int = self._thomas_algorithm(lhs_lower, lhs_main, lhs_upper, rhs_vec)

            # Mettre à jour V
            V[0] = bc_min
            V[1:self.N_space] = V_new_int
            V[self.N_space] = bc_max

            # Exercice anticipé (option américaine)
            if self.exercise == 'american':
                V = np.maximum(V, intrinsic)

        # Interpoler le prix à S0
        price = np.interp(np.log(self.S0), self.x, V)
        return float(price)

    def price(self) -> float:
        """Alias pour solve()."""
        return self.solve()

    def convergence_analysis(self, grid_sizes: Optional[list] = None):
        """
        Analyse de convergence : prix en fonction du raffinement de grille.

        Paramètres:
        -----------
        grid_sizes : list, optional
            Liste de tailles N (utilisées pour N_space et N_time)

        Retourne:
        ---------
        list[dict] : résultats avec N_space, N_time, price, error_vs_bs
        """
        if grid_sizes is None:
            grid_sizes = [25, 50, 100, 200, 400]

        bs_price = black_scholes_price(self.S0, self.K, self.T, self.r, self.sigma, self.option_type)

        results = []
        for n in grid_sizes:
            solver = PDESolver(
                self.S0, self.K, self.T, self.r, self.sigma,
                option_type=self.option_type,
                exercise=self.exercise,
                N_space=n,
                N_time=n,
                S_max_mult=self.S_max_mult,
            )
            pde_price = solver.price()
            error = abs(pde_price - bs_price)
            results.append({
                'N_space': n,
                'N_time': n,
                'price': pde_price,
                'error_vs_bs': error,
            })
            logger.info(f"N={n}: PDE={pde_price:.6f}, BS={bs_price:.6f}, erreur={error:.6f}")

        return results


# ======================================================================
# Fonctions de comparaison
# ======================================================================

def compare_pricing_methods(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call',
) -> dict:
    """
    Compare 4 méthodes de pricing pour une option européenne.

    Méthodes:
        1. Black-Scholes analytique
        2. Monte Carlo (variables antithétiques)
        3. Arbre binomial CRR
        4. Crank-Nicolson PDE

    Paramètres:
    -----------
    S0, K, T, r, sigma : paramètres BSM standard
    option_type : 'call' ou 'put'

    Retourne:
    ---------
    dict : prix par méthode et écarts par rapport à BS
    """
    # 1. Black-Scholes analytique
    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type)

    # 2. Monte Carlo (variables antithétiques)
    n_paths = 100_000
    n_steps = 252
    np.random.seed(42)
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=42)
    S_T = paths[:, -1]
    # Variables antithétiques
    paths_anti = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=43)
    S_T_anti = paths_anti[:, -1]

    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0.0)
        payoffs_anti = np.maximum(S_T_anti - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
        payoffs_anti = np.maximum(K - S_T_anti, 0.0)

    mc_payoffs = 0.5 * (payoffs + payoffs_anti)
    mc_price = float(np.exp(-r * T) * np.mean(mc_payoffs))

    # 3. Arbre binomial CRR
    crr = AmericanOption(S0, K, T, r, sigma, option_type, n_steps=500)
    # Pour obtenir le prix européen via CRR, on utilise l'arbre mais sans exercice anticipé
    # Comme AmericanOption inclut l'exercice anticipé, pour un call sans dividende c'est le même
    # Pour la comparaison, on utilise la valeur CRR directement (call US = call EU sans dividende)
    crr_price = _crr_european_price(S0, K, T, r, sigma, option_type, n_steps=500)

    # 4. PDE Crank-Nicolson
    pde = PDESolver(S0, K, T, r, sigma, option_type=option_type, exercise='european',
                    N_space=300, N_time=300)
    pde_price = pde.price()

    return {
        'black_scholes': bs_price,
        'monte_carlo': mc_price,
        'crr_binomial': crr_price,
        'pde_crank_nicolson': pde_price,
        'diff_mc_vs_bs': mc_price - bs_price,
        'diff_crr_vs_bs': crr_price - bs_price,
        'diff_pde_vs_bs': pde_price - bs_price,
    }


def american_option_comparison(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'put',
) -> dict:
    """
    Compare PDE vs CRR pour option américaine.

    Montre que le PDE capture la prime d'exercice anticipé.

    Paramètres:
    -----------
    S0, K, T, r, sigma : paramètres BSM standard
    option_type : 'call' ou 'put'

    Retourne:
    ---------
    dict : european_bs, american_pde, american_crr, early_exercise_premium
    """
    # Prix européen (référence)
    euro_bs = black_scholes_price(S0, K, T, r, sigma, option_type)

    # PDE américain
    pde_amer = PDESolver(S0, K, T, r, sigma, option_type=option_type, exercise='american',
                         N_space=300, N_time=300)
    amer_pde_price = pde_amer.price()

    # CRR américain
    crr_amer = AmericanOption(S0, K, T, r, sigma, option_type, n_steps=500)
    amer_crr_price = crr_amer.price()

    return {
        'european_bs': euro_bs,
        'american_pde': amer_pde_price,
        'american_crr': amer_crr_price,
        'early_exercise_premium': amer_pde_price - euro_bs,
        'diff_pde_vs_crr': amer_pde_price - amer_crr_price,
    }


def _crr_european_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    n_steps: int = 200,
) -> float:
    """
    Prix européen par arbre binomial CRR (sans exercice anticipé).

    Paramètres:
    -----------
    S0, K, T, r, sigma, option_type : paramètres standard
    n_steps : nombre de pas

    Retourne:
    ---------
    float : prix de l'option européenne
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Payoffs à maturité
    stock_prices = S0 * (u ** np.arange(n_steps + 1)) * (d ** np.arange(n_steps, -1, -1))
    if option_type == 'call':
        option_values = np.maximum(stock_prices - K, 0.0)
    else:
        option_values = np.maximum(K - stock_prices, 0.0)

    # Backward induction (européen : pas de comparaison avec exercice anticipé)
    for i in range(n_steps - 1, -1, -1):
        option_values = discount * (p * option_values[1:i + 2] + (1 - p) * option_values[:i + 1])

    return float(option_values[0])

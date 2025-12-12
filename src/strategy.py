# -*- coding: utf-8 -*-
"""
Module de Stratégies Optionnelles
==================================
Construction et analyse de stratégies multi-legs sur options.

Fournit StrategyBuilder pour assembler des legs arbitraires,
ainsi que des fonctions factory pour les stratégies classiques
(straddle, strangle, bull/bear spreads, butterfly, calendar spread).
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from src.utils import d1, d2, black_scholes_price


# ---------------------------------------------------------------------------
# Black-Scholes Greeks (single option)
# ---------------------------------------------------------------------------

def _bs_delta(S, K, T, r, sigma, option_type='call'):
    """
    Delta Black-Scholes.
        Call: N(d1)
        Put:  N(d1) - 1
    """
    d_1 = d1(S, K, T, r, sigma)
    if option_type == 'call':
        return norm.cdf(d_1)
    else:
        return norm.cdf(d_1) - 1.0


def _bs_gamma(S, K, T, r, sigma):
    """
    Gamma Black-Scholes (same for call and put).
        Gamma = phi(d1) / (S * sigma * sqrt(T))
    """
    if T <= 0:
        return 0.0
    d_1 = d1(S, K, T, r, sigma)
    return norm.pdf(d_1) / (S * sigma * np.sqrt(T))


def _bs_vega(S, K, T, r, sigma):
    """
    Vega Black-Scholes (same for call and put).
        Vega = S * sqrt(T) * phi(d1)

    Returns Vega per 1% move in sigma (i.e., divided by 100).
    """
    if T <= 0:
        return 0.0
    d_1 = d1(S, K, T, r, sigma)
    return S * np.sqrt(T) * norm.pdf(d_1) / 100.0


def _bs_theta(S, K, T, r, sigma, option_type='call'):
    """
    Theta Black-Scholes (per year).
        Call: -(S * phi(d1) * sigma) / (2*sqrt(T)) - r*K*e^(-rT)*N(d2)
        Put:  -(S * phi(d1) * sigma) / (2*sqrt(T)) + r*K*e^(-rT)*N(-d2)
    """
    if T <= 0:
        return 0.0
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    common = -(S * norm.pdf(d_1) * sigma) / (2.0 * np.sqrt(T))
    if option_type == 'call':
        return common - r * K * np.exp(-r * T) * norm.cdf(d_2)
    else:
        return common + r * K * np.exp(-r * T) * norm.cdf(-d_2)


# ---------------------------------------------------------------------------
# OptionLeg dataclass
# ---------------------------------------------------------------------------

class OptionLeg:
    """
    Single option leg in a strategy.

    Attributes
    ----------
    option_type : str  ('call' or 'put')
    strike : float
    maturity : float  (years)
    quantity : int  (number of contracts)
    position : str  ('long' or 'short')
    """

    def __init__(self, option_type: str, strike: float, maturity: float,
                 quantity: int = 1, position: str = 'long'):
        if option_type not in ('call', 'put'):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if position not in ('long', 'short'):
            raise ValueError(f"position must be 'long' or 'short', got '{position}'")
        self.option_type = option_type
        self.strike = strike
        self.maturity = maturity
        self.quantity = quantity
        self.position = position

    @property
    def sign(self) -> int:
        """Return +1 for long, -1 for short."""
        return 1 if self.position == 'long' else -1


# ---------------------------------------------------------------------------
# StrategyBuilder
# ---------------------------------------------------------------------------

class StrategyBuilder:
    """
    Build and analyze multi-leg option strategies.

    Parameters
    ----------
    S : float   — spot price
    r : float   — risk-free rate
    sigma : float — implied volatility
    """

    def __init__(self, S: float, r: float, sigma: float):
        self.S = S
        self.r = r
        self.sigma = sigma
        self.legs: list[OptionLeg] = []

    # ----- leg management -----

    def add_leg(self, option_type: str, strike: float, maturity: float,
                quantity: int = 1, position: str = 'long'):
        """Add an option leg to the strategy."""
        leg = OptionLeg(option_type, strike, maturity, quantity, position)
        self.legs.append(leg)

    # ----- helpers -----

    def _default_spot_range(self) -> np.ndarray:
        """Generate a sensible default spot range around S."""
        return np.linspace(self.S * 0.5, self.S * 1.5, 500)

    @staticmethod
    def _single_payoff(spot: np.ndarray, leg: OptionLeg) -> np.ndarray:
        """Intrinsic payoff of one leg at expiry (per unit), including sign."""
        if leg.option_type == 'call':
            raw = np.maximum(spot - leg.strike, 0.0)
        else:
            raw = np.maximum(leg.strike - spot, 0.0)
        return leg.sign * leg.quantity * raw

    # ----- pricing / premium -----

    def _leg_price(self, leg: OptionLeg) -> float:
        """BS price of one leg (unsigned, per unit)."""
        return black_scholes_price(self.S, leg.strike, leg.maturity,
                                   self.r, self.sigma, leg.option_type)

    def net_premium(self) -> float:
        """
        Total premium paid (positive) or received (negative).
        Long legs cost money (positive), short legs bring money (negative).
        """
        total = 0.0
        for leg in self.legs:
            total += leg.sign * leg.quantity * self._leg_price(leg)
        return total

    # ----- payoff at expiry -----

    def payoff_at_expiry(self, spot_range: np.ndarray | None = None) -> dict:
        """
        Compute combined payoff at expiry (net of premium) for a range of spots.

        Returns
        -------
        dict with 'spots' (np.ndarray) and 'payoff' (np.ndarray)
        """
        if spot_range is None:
            spot_range = self._default_spot_range()
        spots = np.asarray(spot_range, dtype=float)

        payoff = np.zeros_like(spots)
        for leg in self.legs:
            payoff += self._single_payoff(spots, leg)

        # Subtract net premium paid
        payoff -= self.net_premium()

        return {'spots': spots, 'payoff': payoff}

    # ----- Greeks profile -----

    def greeks_profile(self, spot_range: np.ndarray | None = None) -> dict:
        """
        Compute aggregate Greeks as functions of spot price.

        Returns
        -------
        dict with 'spots', 'delta', 'gamma', 'vega', 'theta' arrays
        """
        if spot_range is None:
            spot_range = self._default_spot_range()
        spots = np.asarray(spot_range, dtype=float)

        delta = np.zeros_like(spots)
        gamma = np.zeros_like(spots)
        vega = np.zeros_like(spots)
        theta = np.zeros_like(spots)

        for leg in self.legs:
            w = leg.sign * leg.quantity
            for i, s in enumerate(spots):
                delta[i] += w * _bs_delta(s, leg.strike, leg.maturity,
                                          self.r, self.sigma, leg.option_type)
                gamma[i] += w * _bs_gamma(s, leg.strike, leg.maturity,
                                          self.r, self.sigma)
                vega[i] += w * _bs_vega(s, leg.strike, leg.maturity,
                                        self.r, self.sigma)
                theta[i] += w * _bs_theta(s, leg.strike, leg.maturity,
                                          self.r, self.sigma, leg.option_type)

        return {
            'spots': spots,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
        }

    # ----- breakeven points -----

    def breakeven_points(self) -> list[float]:
        """
        Find spot prices where payoff at expiry = 0 (net of premium).
        Uses numerical root-finding on the payoff function.

        Returns
        -------
        list of float — breakeven spot prices, sorted ascending
        """
        spots = np.linspace(self.S * 0.01, self.S * 3.0, 5000)
        result = self.payoff_at_expiry(spot_range=spots)
        payoff = result['payoff']

        # Find sign changes
        breakevens = []
        for i in range(len(payoff) - 1):
            if payoff[i] * payoff[i + 1] < 0:
                # Refine with Brent's method
                def f(s):
                    p = self.payoff_at_expiry(spot_range=np.array([s]))
                    return p['payoff'][0]
                try:
                    root = brentq(f, spots[i], spots[i + 1], xtol=1e-8)
                    breakevens.append(root)
                except ValueError:
                    pass

        return sorted(breakevens)

    # ----- max profit / loss -----

    def max_profit(self, spot_range: np.ndarray | None = None) -> float:
        """Maximum profit over spot range."""
        if spot_range is None:
            spot_range = np.linspace(self.S * 0.01, self.S * 3.0, 5000)
        result = self.payoff_at_expiry(spot_range=spot_range)
        return float(np.max(result['payoff']))

    def max_loss(self, spot_range: np.ndarray | None = None) -> float:
        """Maximum loss over spot range (negative value)."""
        if spot_range is None:
            spot_range = np.linspace(self.S * 0.01, self.S * 3.0, 5000)
        result = self.payoff_at_expiry(spot_range=spot_range)
        return float(np.min(result['payoff']))


# ===========================================================================
# Factory functions for classic strategies
# ===========================================================================

def straddle(S: float, r: float, sigma: float,
             K: float, T: float, position: str = 'long') -> StrategyBuilder:
    """
    Long/Short Straddle: Call + Put at same strike.

    Long straddle: buy both call and put → profit from large moves.
    Short straddle: sell both → profit if underlying stays near K.
    """
    sb = StrategyBuilder(S, r, sigma)
    sb.add_leg('call', K, T, quantity=1, position=position)
    sb.add_leg('put', K, T, quantity=1, position=position)
    return sb


def strangle(S: float, r: float, sigma: float,
             K_put: float, K_call: float, T: float,
             position: str = 'long') -> StrategyBuilder:
    """
    Long/Short Strangle: OTM Put + OTM Call.

    K_put < K_call.  Cheaper than straddle, needs larger move to profit.
    """
    sb = StrategyBuilder(S, r, sigma)
    sb.add_leg('put', K_put, T, quantity=1, position=position)
    sb.add_leg('call', K_call, T, quantity=1, position=position)
    return sb


def bull_call_spread(S: float, r: float, sigma: float,
                     K_low: float, K_high: float, T: float) -> StrategyBuilder:
    """
    Bull Call Spread: Long Call(K_low) + Short Call(K_high).

    Bullish strategy with capped profit and limited loss.
    """
    sb = StrategyBuilder(S, r, sigma)
    sb.add_leg('call', K_low, T, quantity=1, position='long')
    sb.add_leg('call', K_high, T, quantity=1, position='short')
    return sb


def bear_put_spread(S: float, r: float, sigma: float,
                    K_low: float, K_high: float, T: float) -> StrategyBuilder:
    """
    Bear Put Spread: Long Put(K_high) + Short Put(K_low).

    Bearish strategy with capped profit and limited loss.
    """
    sb = StrategyBuilder(S, r, sigma)
    sb.add_leg('put', K_high, T, quantity=1, position='long')
    sb.add_leg('put', K_low, T, quantity=1, position='short')
    return sb


def butterfly(S: float, r: float, sigma: float,
              K_low: float, K_mid: float, K_high: float,
              T: float) -> StrategyBuilder:
    """
    Long Butterfly: Long Call(K_low) + 2x Short Call(K_mid) + Long Call(K_high).

    Profits when underlying stays near K_mid at expiry.
    Requires K_high - K_mid = K_mid - K_low (equidistant strikes).
    """
    sb = StrategyBuilder(S, r, sigma)
    sb.add_leg('call', K_low, T, quantity=1, position='long')
    sb.add_leg('call', K_mid, T, quantity=2, position='short')
    sb.add_leg('call', K_high, T, quantity=1, position='long')
    return sb


def calendar_spread(S: float, r: float, sigma: float,
                    K: float, T_short: float, T_long: float) -> StrategyBuilder:
    """
    Calendar Spread: Short Call(T_short) + Long Call(T_long). Same strike.

    Profits from time-decay differential. Short Gamma, Long Vega at ATM.
    """
    sb = StrategyBuilder(S, r, sigma)
    sb.add_leg('call', K, T_short, quantity=1, position='short')
    sb.add_leg('call', K, T_long, quantity=1, position='long')
    return sb

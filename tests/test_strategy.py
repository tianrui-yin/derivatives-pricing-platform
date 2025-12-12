# -*- coding: utf-8 -*-
"""
Tests for Option Strategy Module
=================================
TDD: write tests first, then implement.
"""

import numpy as np
import pytest

from src.strategy import (
    OptionLeg,
    StrategyBuilder,
    straddle,
    strangle,
    bull_call_spread,
    bear_put_spread,
    butterfly,
    calendar_spread,
)


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def market_params():
    """Standard market parameters for tests."""
    return dict(S=100.0, r=0.05, sigma=0.20)


@pytest.fixture
def default_maturity():
    return 0.5  # 6 months


# ===========================================================================
# 1. StrategyBuilder basics
# ===========================================================================

class TestStrategyBuilder:
    """Tests for StrategyBuilder core functionality."""

    def test_add_leg(self, market_params):
        sb = StrategyBuilder(**market_params)
        sb.add_leg('call', 100, 0.5, quantity=1, position='long')
        assert len(sb.legs) == 1
        assert isinstance(sb.legs[0], OptionLeg)

    def test_add_multiple_legs(self, market_params):
        sb = StrategyBuilder(**market_params)
        sb.add_leg('call', 100, 0.5)
        sb.add_leg('put', 100, 0.5)
        assert len(sb.legs) == 2

    def test_payoff_at_expiry_shape(self, market_params):
        sb = StrategyBuilder(**market_params)
        sb.add_leg('call', 100, 0.5)
        result = sb.payoff_at_expiry()
        assert 'spots' in result
        assert 'payoff' in result
        assert len(result['spots']) == len(result['payoff'])
        assert len(result['spots']) > 0

    def test_payoff_at_expiry_custom_range(self, market_params):
        sb = StrategyBuilder(**market_params)
        sb.add_leg('call', 100, 0.5)
        spots = np.linspace(80, 120, 50)
        result = sb.payoff_at_expiry(spot_range=spots)
        assert len(result['spots']) == 50
        np.testing.assert_array_equal(result['spots'], spots)

    def test_net_premium_long_call(self, market_params):
        sb = StrategyBuilder(**market_params)
        sb.add_leg('call', 100, 0.5)
        premium = sb.net_premium()
        # Long call => pay premium => positive net premium
        assert premium > 0

    def test_net_premium_short_call(self, market_params):
        sb = StrategyBuilder(**market_params)
        sb.add_leg('call', 100, 0.5, position='short')
        premium = sb.net_premium()
        # Short call => receive premium => negative net premium
        assert premium < 0

    def test_add_leg_invalid_option_type(self, market_params):
        sb = StrategyBuilder(**market_params)
        with pytest.raises(ValueError):
            sb.add_leg('forward', 100, 0.5)

    def test_add_leg_invalid_position(self, market_params):
        sb = StrategyBuilder(**market_params)
        with pytest.raises(ValueError):
            sb.add_leg('call', 100, 0.5, position='neutral')


# ===========================================================================
# 2. Straddle
# ===========================================================================

class TestStraddle:
    """Tests for straddle strategy."""

    def test_straddle_has_two_legs(self, market_params):
        sb = straddle(**market_params, K=100, T=0.5)
        assert len(sb.legs) == 2

    def test_straddle_payoff_v_shaped(self, market_params):
        """Straddle payoff is V-shaped: minimum at strike, increases both sides."""
        sb = straddle(**market_params, K=100, T=0.5)
        spots = np.linspace(60, 140, 500)
        result = sb.payoff_at_expiry(spot_range=spots)
        payoff = result['payoff']

        # Find the index closest to K=100
        idx_at_strike = np.argmin(np.abs(spots - 100))

        # Payoff at strike should be the minimum (or very close)
        assert payoff[idx_at_strike] == pytest.approx(np.min(payoff), abs=0.5)

        # Payoff should increase as we move away from strike
        assert payoff[0] > payoff[idx_at_strike]
        assert payoff[-1] > payoff[idx_at_strike]

    def test_straddle_symmetric_around_strike(self, market_params):
        """Straddle payoff is symmetric around K."""
        sb = straddle(**market_params, K=100, T=0.5)
        premium = sb.net_premium()

        # Payoff at K+20 and K-20 should be equal
        spots = np.array([80.0, 120.0])
        result = sb.payoff_at_expiry(spot_range=spots)
        payoff = result['payoff']
        assert payoff[0] == pytest.approx(payoff[1], abs=0.01)

    def test_straddle_max_loss_equals_premium(self, market_params):
        """Long straddle max loss = net premium paid."""
        sb = straddle(**market_params, K=100, T=0.5, position='long')
        premium = sb.net_premium()
        max_loss = sb.max_loss()
        # Tolerance accounts for discrete spot sampling not hitting K exactly
        assert max_loss == pytest.approx(-premium, abs=0.05)

    def test_short_straddle(self, market_params):
        """Short straddle is inverse of long straddle."""
        sb_long = straddle(**market_params, K=100, T=0.5, position='long')
        sb_short = straddle(**market_params, K=100, T=0.5, position='short')

        spots = np.linspace(60, 140, 100)
        payoff_long = sb_long.payoff_at_expiry(spot_range=spots)['payoff']
        payoff_short = sb_short.payoff_at_expiry(spot_range=spots)['payoff']

        np.testing.assert_allclose(payoff_long, -payoff_short, atol=0.01)


# ===========================================================================
# 3. Strangle
# ===========================================================================

class TestStrangle:
    """Tests for strangle strategy."""

    def test_strangle_has_two_legs(self, market_params):
        sb = strangle(**market_params, K_put=90, K_call=110, T=0.5)
        assert len(sb.legs) == 2

    def test_strangle_wider_v_shape_than_straddle(self, market_params):
        """Strangle has a wider flat bottom than straddle."""
        sb_straddle = straddle(**market_params, K=100, T=0.5)
        sb_strangle = strangle(**market_params, K_put=90, K_call=110, T=0.5)

        # Between the two strikes, strangle payoff should be flat (= -premium)
        spots = np.linspace(90, 110, 50)
        payoff = sb_strangle.payoff_at_expiry(spot_range=spots)['payoff']
        # All payoffs between strikes should be equal (flat region)
        np.testing.assert_allclose(payoff, payoff[0], atol=0.01)

    def test_strangle_cheaper_than_straddle(self, market_params):
        """Strangle is cheaper than straddle (OTM options)."""
        sb_straddle = straddle(**market_params, K=100, T=0.5)
        sb_strangle = strangle(**market_params, K_put=90, K_call=110, T=0.5)

        assert sb_strangle.net_premium() < sb_straddle.net_premium()


# ===========================================================================
# 4. Bull Call Spread
# ===========================================================================

class TestBullCallSpread:
    """Tests for bull call spread strategy."""

    def test_bull_call_spread_two_legs(self, market_params):
        sb = bull_call_spread(**market_params, K_low=95, K_high=105, T=0.5)
        assert len(sb.legs) == 2

    def test_bull_call_spread_max_profit(self, market_params):
        """Max profit = K_high - K_low - net_premium."""
        sb = bull_call_spread(**market_params, K_low=95, K_high=105, T=0.5)
        premium = sb.net_premium()
        max_profit = sb.max_profit()
        expected_max_profit = (105 - 95) - premium
        assert max_profit == pytest.approx(expected_max_profit, abs=0.01)

    def test_bull_call_spread_max_loss(self, market_params):
        """Max loss = net premium paid."""
        sb = bull_call_spread(**market_params, K_low=95, K_high=105, T=0.5)
        premium = sb.net_premium()
        max_loss = sb.max_loss()
        assert max_loss == pytest.approx(-premium, abs=0.01)

    def test_bull_call_spread_bounded_payoff(self, market_params):
        """Payoff is bounded between -premium and (K_high - K_low - premium)."""
        sb = bull_call_spread(**market_params, K_low=95, K_high=105, T=0.5)
        premium = sb.net_premium()
        spots = np.linspace(50, 200, 1000)
        payoff = sb.payoff_at_expiry(spot_range=spots)['payoff']

        assert np.min(payoff) >= -premium - 0.01
        assert np.max(payoff) <= (105 - 95) - premium + 0.01


# ===========================================================================
# 5. Bear Put Spread
# ===========================================================================

class TestBearPutSpread:
    """Tests for bear put spread strategy."""

    def test_bear_put_spread_two_legs(self, market_params):
        sb = bear_put_spread(**market_params, K_low=95, K_high=105, T=0.5)
        assert len(sb.legs) == 2

    def test_bear_put_spread_max_profit(self, market_params):
        """Max profit = K_high - K_low - net_premium."""
        sb = bear_put_spread(**market_params, K_low=95, K_high=105, T=0.5)
        premium = sb.net_premium()
        max_profit = sb.max_profit()
        expected_max_profit = (105 - 95) - premium
        assert max_profit == pytest.approx(expected_max_profit, abs=0.01)

    def test_bear_put_spread_max_loss(self, market_params):
        """Max loss = net premium paid."""
        sb = bear_put_spread(**market_params, K_low=95, K_high=105, T=0.5)
        premium = sb.net_premium()
        max_loss = sb.max_loss()
        assert max_loss == pytest.approx(-premium, abs=0.01)

    def test_bear_put_spread_bounded_payoff(self, market_params):
        """Payoff is bounded."""
        sb = bear_put_spread(**market_params, K_low=95, K_high=105, T=0.5)
        premium = sb.net_premium()
        spots = np.linspace(50, 200, 1000)
        payoff = sb.payoff_at_expiry(spot_range=spots)['payoff']

        assert np.min(payoff) >= -premium - 0.01
        assert np.max(payoff) <= (105 - 95) - premium + 0.01


# ===========================================================================
# 6. Butterfly
# ===========================================================================

class TestButterfly:
    """Tests for butterfly spread strategy."""

    def test_butterfly_four_legs(self, market_params):
        """Butterfly: long 1 call(K_low) + short 2 calls(K_mid) + long 1 call(K_high)."""
        sb = butterfly(**market_params, K_low=90, K_mid=100, K_high=110, T=0.5)
        assert len(sb.legs) == 3  # 3 distinct legs (one with quantity=2)

    def test_butterfly_max_profit_at_mid_strike(self, market_params):
        """Maximum profit occurs at K_mid."""
        sb = butterfly(**market_params, K_low=90, K_mid=100, K_high=110, T=0.5)
        spots = np.linspace(60, 140, 1000)
        result = sb.payoff_at_expiry(spot_range=spots)
        payoff = result['payoff']

        idx_max = np.argmax(payoff)
        assert result['spots'][idx_max] == pytest.approx(100.0, abs=0.5)

    def test_butterfly_max_loss_equals_premium(self, market_params):
        """Maximum loss = net premium paid."""
        sb = butterfly(**market_params, K_low=90, K_mid=100, K_high=110, T=0.5)
        premium = sb.net_premium()
        max_loss = sb.max_loss()
        assert max_loss == pytest.approx(-premium, abs=0.01)

    def test_butterfly_payoff_not_below_negative_premium(self, market_params):
        """Payoff >= -premium everywhere."""
        sb = butterfly(**market_params, K_low=90, K_mid=100, K_high=110, T=0.5)
        premium = sb.net_premium()
        spots = np.linspace(50, 150, 1000)
        payoff = sb.payoff_at_expiry(spot_range=spots)['payoff']
        assert np.min(payoff) >= -premium - 0.01


# ===========================================================================
# 7. Calendar Spread — Short Gamma + Long Vega (GS interview)
# ===========================================================================

class TestCalendarSpread:
    """Tests for calendar spread including Gamma/Vega properties."""

    def test_calendar_spread_two_legs(self, market_params):
        sb = calendar_spread(**market_params, K=100, T_short=0.25, T_long=0.75)
        assert len(sb.legs) == 2

    def test_calendar_spread_short_gamma_at_atm(self, market_params):
        """
        Calendar spread (short near, long far) has NEGATIVE aggregate Gamma
        near ATM, because the short near-term option has higher Gamma than
        the long far-term option.
        """
        sb = calendar_spread(**market_params, K=100, T_short=0.25, T_long=0.75)
        spots = np.array([100.0])
        greeks = sb.greeks_profile(spot_range=spots)
        gamma_atm = greeks['gamma'][0]
        assert gamma_atm < 0, f"Expected negative Gamma at ATM, got {gamma_atm}"

    def test_calendar_spread_long_vega_at_atm(self, market_params):
        """
        Calendar spread (short near, long far) has POSITIVE aggregate Vega
        near ATM, because the long far-term option has higher Vega than
        the short near-term option.
        """
        sb = calendar_spread(**market_params, K=100, T_short=0.25, T_long=0.75)
        spots = np.array([100.0])
        greeks = sb.greeks_profile(spot_range=spots)
        vega_atm = greeks['vega'][0]
        assert vega_atm > 0, f"Expected positive Vega at ATM, got {vega_atm}"

    def test_calendar_spread_gamma_vega_combined(self, market_params):
        """Combined test: short Gamma AND long Vega at ATM."""
        sb = calendar_spread(**market_params, K=100, T_short=0.25, T_long=0.75)
        spots = np.array([100.0])
        greeks = sb.greeks_profile(spot_range=spots)
        assert greeks['gamma'][0] < 0, "Calendar spread should be short Gamma"
        assert greeks['vega'][0] > 0, "Calendar spread should be long Vega"


# ===========================================================================
# 8. Greeks profile
# ===========================================================================

class TestGreeksProfile:
    """Tests for Greeks profile computation."""

    def test_greeks_profile_keys(self, market_params):
        sb = StrategyBuilder(**market_params)
        sb.add_leg('call', 100, 0.5)
        result = sb.greeks_profile()
        for key in ['spots', 'delta', 'gamma', 'vega', 'theta']:
            assert key in result

    def test_long_call_delta_range(self, market_params):
        """Delta of a long call is in (0, 1)."""
        sb = StrategyBuilder(**market_params)
        sb.add_leg('call', 100, 0.5)
        spots = np.linspace(60, 140, 200)
        greeks = sb.greeks_profile(spot_range=spots)
        delta = greeks['delta']
        assert np.all(delta > -0.01)  # small tolerance
        assert np.all(delta < 1.01)

    def test_long_call_gamma_positive(self, market_params):
        """Gamma of a long call is always positive."""
        sb = StrategyBuilder(**market_params)
        sb.add_leg('call', 100, 0.5)
        spots = np.linspace(60, 140, 200)
        greeks = sb.greeks_profile(spot_range=spots)
        gamma = greeks['gamma']
        assert np.all(gamma >= -1e-10)  # allow tiny numerical noise

    def test_long_put_delta_range(self, market_params):
        """Delta of a long put is in (-1, 0)."""
        sb = StrategyBuilder(**market_params)
        sb.add_leg('put', 100, 0.5)
        spots = np.linspace(60, 140, 200)
        greeks = sb.greeks_profile(spot_range=spots)
        delta = greeks['delta']
        assert np.all(delta > -1.01)
        assert np.all(delta < 0.01)

    def test_short_call_delta_negative(self, market_params):
        """Delta of a short call is in (-1, 0)."""
        sb = StrategyBuilder(**market_params)
        sb.add_leg('call', 100, 0.5, position='short')
        spots = np.linspace(60, 140, 200)
        greeks = sb.greeks_profile(spot_range=spots)
        delta = greeks['delta']
        assert np.all(delta > -1.01)
        assert np.all(delta < 0.01)

    def test_straddle_delta_near_zero_atm(self, market_params):
        """Long straddle Delta is near zero at ATM (call delta ~ 0.5, put delta ~ -0.5)."""
        sb = straddle(**market_params, K=100, T=0.5)
        spots = np.array([100.0])
        greeks = sb.greeks_profile(spot_range=spots)
        # Delta should be close to 0 at ATM (not exactly 0 because of r > 0
        # shifts the call delta above 0.5 and put delta above -0.5)
        assert abs(greeks['delta'][0]) < 0.25

    def test_greeks_profile_shape(self, market_params):
        """All Greek arrays should have the same length as spots."""
        sb = StrategyBuilder(**market_params)
        sb.add_leg('call', 100, 0.5)
        spots = np.linspace(80, 120, 30)
        greeks = sb.greeks_profile(spot_range=spots)
        for key in ['delta', 'gamma', 'vega', 'theta']:
            assert len(greeks[key]) == 30


# ===========================================================================
# 9. Breakeven points
# ===========================================================================

class TestBreakevenPoints:
    """Tests for breakeven point computation."""

    def test_straddle_two_breakevens(self, market_params):
        """Long straddle has exactly 2 breakeven points."""
        sb = straddle(**market_params, K=100, T=0.5)
        bep = sb.breakeven_points()
        assert len(bep) == 2

    def test_straddle_breakeven_symmetric(self, market_params):
        """Breakeven points should be roughly symmetric around K."""
        sb = straddle(**market_params, K=100, T=0.5)
        bep = sorted(sb.breakeven_points())
        premium = sb.net_premium()
        # Lower breakeven ~ K - premium, upper ~ K + premium
        assert bep[0] == pytest.approx(100 - premium, abs=0.5)
        assert bep[1] == pytest.approx(100 + premium, abs=0.5)

    def test_bull_call_spread_one_breakeven(self, market_params):
        """Bull call spread has 1 breakeven point between K_low and K_high."""
        sb = bull_call_spread(**market_params, K_low=95, K_high=105, T=0.5)
        bep = sb.breakeven_points()
        assert len(bep) == 1
        assert 95 < bep[0] < 105

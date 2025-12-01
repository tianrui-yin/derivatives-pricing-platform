# -*- coding: utf-8 -*-
"""
Tests Unitaires pour la Plateforme de Pricing de Dérivés
========================================================
Tests basiques pour valider les différents produits.
"""

import sys
import os
import numpy as np

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vanilla_options import EuropeanOption, AmericanOption
from src.barrier_options import BarrierOption
from src.asian_options import AsianOption
from src.lookback_options import LookbackOption
from src.interest_rate import InterestRateSwap, Cap, Floor
from src.repo_forward import Repo, Forward


def test_european_call_put_parity():
    """
    Test de la parité Put-Call pour options européennes.

    Formule: c + K*e^(-rT) = p + S
    """
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    call = EuropeanOption(S, K, T, r, sigma, 'call')
    put = EuropeanOption(S, K, T, r, sigma, 'put')

    call_price = call.price()
    put_price = put.price()

    left_side = call_price + K * np.exp(-r * T)
    right_side = put_price + S

    assert abs(left_side - right_side) < 0.01, \
        f"Parité Put-Call violée: {left_side:.4f} ≠ {right_side:.4f}"

    print(f"✓ Parité Put-Call: écart = {abs(left_side - right_side):.6f}")


def test_american_vs_european_call():
    """
    Test: Call américain = Call européen (sans dividende).

    Sans dividende, il n'est jamais optimal d'exercer un call avant maturité.
    """
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    euro_call = EuropeanOption(S, K, T, r, sigma, 'call')
    amer_call = AmericanOption(S, K, T, r, sigma, 'call', n_steps=200)

    euro_price = euro_call.price()
    amer_price = amer_call.price()

    # Le call américain devrait être très proche du call européen
    assert abs(amer_price - euro_price) < 0.05, \
        f"Call US ({amer_price:.4f}) ≠ Call EU ({euro_price:.4f})"

    print(f"✓ Call US ≈ Call EU: {amer_price:.4f} ≈ {euro_price:.4f}")


def test_american_put_premium():
    """
    Test: Put américain >= Put européen.

    L'exercice anticipé peut être optimal pour un put.
    """
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    euro_put = EuropeanOption(S, K, T, r, sigma, 'put')
    amer_put = AmericanOption(S, K, T, r, sigma, 'put', n_steps=200)

    euro_price = euro_put.price()
    amer_price = amer_put.price()

    assert amer_price >= euro_price - 0.01, \
        f"Put US ({amer_price:.4f}) < Put EU ({euro_price:.4f})"

    premium = amer_price - euro_price
    print(f"✓ Put US >= Put EU: prime = {premium:.4f}")


def test_barrier_in_out_parity():
    """
    Test: Knock-In + Knock-Out = Vanille.
    """
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25
    barrier = 80

    vanilla = EuropeanOption(S, K, T, r, sigma, 'call')
    dao = BarrierOption(S, K, T, r, sigma, barrier, 'down-and-out', 'call')
    dai = BarrierOption(S, K, T, r, sigma, barrier, 'down-and-in', 'call')

    vanilla_price = vanilla.price()
    sum_price = dao.price() + dai.price()

    # Tolérance plus large pour Monte Carlo
    tolerance = vanilla_price * 0.10  # 10%
    assert abs(sum_price - vanilla_price) < tolerance, \
        f"In + Out ({sum_price:.4f}) ≠ Vanille ({vanilla_price:.4f})"

    print(f"✓ Parité In-Out: {sum_price:.4f} ≈ {vanilla_price:.4f}")


def test_asian_cheaper_than_vanilla():
    """
    Test: Option asiatique < Option vanille.

    La moyenne réduit la volatilité effective.
    """
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25

    vanilla = EuropeanOption(S, K, T, r, sigma, 'call')
    asian = AsianOption(S, K, T, r, sigma, 'arithmetic', 'call')

    vanilla_price = vanilla.price()
    asian_price = asian.price()

    assert asian_price < vanilla_price, \
        f"Asian ({asian_price:.4f}) >= Vanille ({vanilla_price:.4f})"

    discount = asian_price / vanilla_price
    print(f"✓ Asian < Vanille: {asian_price:.4f} < {vanilla_price:.4f} ({discount:.1%})")


def test_lookback_more_expensive():
    """
    Test: Option lookback > Option vanille.

    Le lookback élimine le risque de mauvais timing.
    """
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25

    vanilla = EuropeanOption(S, K, T, r, sigma, 'call')
    lookback = LookbackOption(S, K, T, r, sigma, 'floating', 'call')

    vanilla_price = vanilla.price()
    lookback_price = lookback.price()

    # Le lookback devrait être significativement plus cher
    assert lookback_price > vanilla_price, \
        f"Lookback ({lookback_price:.4f}) <= Vanille ({vanilla_price:.4f})"

    premium = lookback_price / vanilla_price
    print(f"✓ Lookback > Vanille: {lookback_price:.4f} ({premium:.1f}x)")


def test_swap_par_rate():
    """
    Test: Un swap au taux par a une valeur proche de 0.
    """
    notional = 10_000_000
    swap = InterestRateSwap(notional, fixed_rate=0.025, maturity=5)
    swap._generate_default_curve(0.025)  # Courbe plate au même taux

    par_rate = swap.par_rate()
    swap.fixed_rate = par_rate
    value = swap.price()

    # La valeur devrait être proche de 0
    assert abs(value) < notional * 0.001, \
        f"Valeur du swap par ({value:,.0f}) trop élevée"

    print(f"✓ Swap par rate: valeur = {value:,.2f} € (≈ 0)")


def test_forward_price():
    """
    Test du prix forward théorique.

    Formule: F = S * e^((r-q)T)
    """
    S, r, T, q = 100, 0.05, 1, 0.02

    forward = Forward(S, r, T, q)
    F = forward.forward_price()

    expected = S * np.exp((r - q) * T)

    assert abs(F - expected) < 0.001, \
        f"Forward ({F:.4f}) ≠ attendu ({expected:.4f})"

    print(f"✓ Prix Forward: {F:.4f} = {expected:.4f}")


def test_forward_value_at_inception():
    """
    Test: Valeur du forward = 0 à l'initiation.
    """
    S, r, T, q = 100, 0.05, 1, 0.02

    forward = Forward(S, r, T, q)  # K = F automatiquement
    value = forward.price('long')

    assert abs(value) < 0.001, \
        f"Valeur initiale ({value:.4f}) ≠ 0"

    print(f"✓ Forward valeur initiale: {value:.6f} ≈ 0")


def test_repo_interest():
    """
    Test du calcul des intérêts repo.
    """
    principal = 10_000_000
    repo_rate = 0.02  # 2%
    days = 30

    repo = Repo(principal, repo_rate, days / 365)

    expected_interest = principal * repo_rate * (days / 365)
    actual_interest = repo.interest_amount()

    assert abs(actual_interest - expected_interest) < 1, \
        f"Intérêts ({actual_interest:.2f}) ≠ attendu ({expected_interest:.2f})"

    print(f"✓ Intérêts Repo: {actual_interest:,.2f} €")


def test_cap_floor_positive():
    """
    Test que Cap et Floor ont des prix positifs.
    """
    notional = 10_000_000
    strike = 0.025
    maturity = 5
    vol = 0.20

    cap = Cap(notional, strike, maturity, vol)
    floor = Floor(notional, strike, maturity, vol)
    cap._generate_default_market()
    floor._generate_default_market()

    cap_price = cap.price()
    floor_price = floor.price()

    assert cap_price >= 0, f"Prix Cap négatif: {cap_price}"
    assert floor_price >= 0, f"Prix Floor négatif: {floor_price}"

    print(f"✓ Cap/Floor positifs: Cap={cap_price:,.0f}€, Floor={floor_price:,.0f}€")


def run_all_tests():
    """
    Exécute tous les tests unitaires.
    """
    print("=" * 60)
    print("TESTS UNITAIRES - Plateforme de Pricing de Dérivés")
    print("=" * 60)
    print()

    tests = [
        test_european_call_put_parity,
        test_american_vs_european_call,
        test_american_put_premium,
        test_barrier_in_out_parity,
        test_asian_cheaper_than_vanilla,
        test_lookback_more_expensive,
        test_swap_par_rate,
        test_forward_price,
        test_forward_value_at_inception,
        test_repo_interest,
        test_cap_floor_positive
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} ÉCHEC: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERREUR: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Résultats: {passed} réussis, {failed} échoués sur {len(tests)}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

# -*- coding: utf-8 -*-
"""
Démonstration de la Plateforme de Pricing de Dérivés
====================================================
Script complet démontrant les 15+ produits implémentés.
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


def demo_vanilla_options():
    """
    Démonstration 1: Options Vanilles.
    """
    print("\n" + "=" * 70)
    print("DÉMONSTRATION 1: OPTIONS VANILLES")
    print("=" * 70)

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20

    print(f"\nParamètres: S={S}, K={K}, T={T}an, r={r:.1%}, σ={sigma:.1%}")

    # Options européennes
    print(f"\n--- Options Européennes (Black-Scholes) ---")
    euro_call = EuropeanOption(S, K, T, r, sigma, 'call')
    euro_put = EuropeanOption(S, K, T, r, sigma, 'put')

    print(f"Call européen: {euro_call.price():.4f} €")
    print(f"  Delta: {euro_call.delta():.4f}")
    print(f"  Gamma: {euro_call.gamma():.6f}")
    print(f"  Vega:  {euro_call.vega():.4f}")
    print(f"Put européen:  {euro_put.price():.4f} €")

    # Options américaines
    print(f"\n--- Options Américaines (Arbre Binomial) ---")
    amer_call = AmericanOption(S, K, T, r, sigma, 'call', n_steps=200)
    amer_put = AmericanOption(S, K, T, r, sigma, 'put', n_steps=200)

    print(f"Call américain: {amer_call.price():.4f} €")
    print(f"Put américain:  {amer_put.price():.4f} €")
    print(f"Prime exercice anticipé (put): {amer_put.early_exercise_premium():.4f} €")


def demo_barrier_options():
    """
    Démonstration 2: Options Barrières.
    """
    print("\n" + "=" * 70)
    print("DÉMONSTRATION 2: OPTIONS BARRIÈRES")
    print("=" * 70)

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25

    # Vanille pour comparaison
    vanilla_call = EuropeanOption(S, K, T, r, sigma, 'call')
    print(f"\nCall vanille (référence): {vanilla_call.price():.4f} €")

    # Down barriers
    print(f"\n--- Barrières DOWN (B=80) ---")
    barrier_down = 80
    dao_call = BarrierOption(S, K, T, r, sigma, barrier_down, 'down-and-out', 'call')
    dai_call = BarrierOption(S, K, T, r, sigma, barrier_down, 'down-and-in', 'call')

    print(f"Down-and-Out Call: {dao_call.price():.4f} €")
    print(f"Down-and-In Call:  {dai_call.price():.4f} €")
    print(f"Somme (≈vanille):  {dao_call.price() + dai_call.price():.4f} €")
    print(f"P(toucher B=80):   {dao_call.barrier_probability():.1%}")

    # Up barriers
    print(f"\n--- Barrières UP (B=120) ---")
    barrier_up = 120
    uao_call = BarrierOption(S, K, T, r, sigma, barrier_up, 'up-and-out', 'call')
    uai_call = BarrierOption(S, K, T, r, sigma, barrier_up, 'up-and-in', 'call')

    print(f"Up-and-Out Call: {uao_call.price():.4f} €")
    print(f"Up-and-In Call:  {uai_call.price():.4f} €")


def demo_asian_options():
    """
    Démonstration 3: Options Asiatiques.
    """
    print("\n" + "=" * 70)
    print("DÉMONSTRATION 3: OPTIONS ASIATIQUES")
    print("=" * 70)

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25

    vanilla = EuropeanOption(S, K, T, r, sigma, 'call')
    asian_arith = AsianOption(S, K, T, r, sigma, 'arithmetic', 'call')
    asian_geom = AsianOption(S, K, T, r, sigma, 'geometric', 'call')

    vanilla_price = vanilla.price()
    arith_price = asian_arith.price()
    geom_price = asian_geom.price()

    print(f"\nComparaison des prix:")
    print(f"  Vanille:      {vanilla_price:.4f} € (100%)")
    print(f"  Arithmétique: {arith_price:.4f} € ({arith_price/vanilla_price:.1%})")
    print(f"  Géométrique:  {geom_price:.4f} € ({geom_price/vanilla_price:.1%})")

    print(f"\n--- Effet du nombre de fixings ---")
    for n in [4, 12, 52, 252]:
        asian = AsianOption(S, K, T, r, sigma, 'arithmetic', 'call', n_fixings=n)
        print(f"  {n:3d} fixings: {asian.price():.4f} €")


def demo_lookback_options():
    """
    Démonstration 4: Options Lookback.
    """
    print("\n" + "=" * 70)
    print("DÉMONSTRATION 4: OPTIONS LOOKBACK")
    print("=" * 70)

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25

    vanilla = EuropeanOption(S, K, T, r, sigma, 'call')
    vanilla_price = vanilla.price()

    print(f"\nVanille (référence): {vanilla_price:.4f} €")

    print(f"\n--- Floating Strike ---")
    lb_float_call = LookbackOption(S, K, T, r, sigma, 'floating', 'call')
    lb_float_put = LookbackOption(S, K, T, r, sigma, 'floating', 'put')

    float_call_price = lb_float_call.price()
    print(f"Call (S_T - S_min): {float_call_price:.4f} € ({float_call_price/vanilla_price:.1f}x vanille)")
    print(f"Put (S_max - S_T):  {lb_float_put.price():.4f} €")

    print(f"\n--- Fixed Strike ---")
    lb_fixed_call = LookbackOption(S, K, T, r, sigma, 'fixed', 'call')
    lb_fixed_put = LookbackOption(S, K, T, r, sigma, 'fixed', 'put')

    fixed_call_price = lb_fixed_call.price()
    print(f"Call (S_max - K)+: {fixed_call_price:.4f} € ({fixed_call_price/vanilla_price:.1f}x vanille)")
    print(f"Put (K - S_min)+:  {lb_fixed_put.price():.4f} €")


def demo_interest_rate_products():
    """
    Démonstration 5: Produits de Taux.
    """
    print("\n" + "=" * 70)
    print("DÉMONSTRATION 5: PRODUITS DE TAUX")
    print("=" * 70)

    notional = 10_000_000  # 10 millions €
    maturity = 5
    flat_rate = 0.025

    # Interest Rate Swap
    print(f"\n--- Interest Rate Swap ---")
    swap = InterestRateSwap(notional, fixed_rate=0.025, maturity=maturity)
    swap._generate_default_curve(flat_rate)

    print(f"Notional: {notional:,.0f} €")
    print(f"Maturité: {maturity} ans")
    print(f"Taux fixe contractuel: {swap.fixed_rate:.2%}")
    print(f"Taux swap (par rate): {swap.par_rate():.4%}")
    print(f"Valeur du swap: {swap.price():,.0f} €")
    print(f"DV01: {swap.dv01():,.0f} €")

    # Cap
    print(f"\n--- Cap ---")
    cap = Cap(notional, strike=0.03, maturity=maturity, vol=0.20)
    cap._generate_default_market(flat_rate)
    print(f"Strike: {cap.strike:.2%}")
    print(f"Volatilité: {cap.vol:.1%}")
    print(f"Prix du Cap: {cap.price():,.0f} €")

    # Floor
    print(f"\n--- Floor ---")
    floor = Floor(notional, strike=0.02, maturity=maturity, vol=0.20)
    floor._generate_default_market(flat_rate)
    print(f"Strike: {floor.strike:.2%}")
    print(f"Prix du Floor: {floor.price():,.0f} €")


def demo_repo_forward():
    """
    Démonstration 6: Repos et Forwards.
    """
    print("\n" + "=" * 70)
    print("DÉMONSTRATION 6: REPOS ET FORWARDS")
    print("=" * 70)

    # Repo
    print(f"\n--- Repo (7 jours) ---")
    principal = 10_000_000
    repo_rate = 0.015

    repo = Repo(principal, repo_rate, 7/365)
    print(f"Principal: {principal:,.0f} €")
    print(f"Taux repo: {repo_rate:.2%}")
    print(f"Intérêts: {repo.interest_amount():,.2f} €")
    print(f"Prix de rachat: {repo.repurchase_price():,.2f} €")

    # Forward
    print(f"\n--- Forward sur Action ---")
    S, r, T, q = 100, 0.05, 1, 0.02

    forward = Forward(S, r, T, q)
    print(f"Prix spot: {S:.2f} €")
    print(f"Taux sans risque: {r:.2%}")
    print(f"Dividende: {q:.2%}")
    print(f"Prix forward: {forward.forward_price():.4f} €")
    print(f"Delta: {forward.delta():.4f}")

    # Payoffs
    print(f"\n--- Payoffs à différents prix finaux ---")
    for S_T in [90, 100, 110]:
        payoff = forward.payoff(S_T, 'long')
        print(f"S_T = {S_T}: Payoff = {payoff:+.2f} €")


def demo_summary():
    """
    Démonstration 7: Résumé des 15+ produits.
    """
    print("\n" + "=" * 70)
    print("RÉSUMÉ: TOUS LES PRODUITS IMPLÉMENTÉS")
    print("=" * 70)

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25

    products = []

    # 1-2: Options européennes
    products.append(('1. Call européen', EuropeanOption(S, K, T, r, sigma, 'call').price()))
    products.append(('2. Put européen', EuropeanOption(S, K, T, r, sigma, 'put').price()))

    # 3-4: Options américaines
    products.append(('3. Call américain', AmericanOption(S, K, T, r, sigma, 'call').price()))
    products.append(('4. Put américain', AmericanOption(S, K, T, r, sigma, 'put').price()))

    # 5-8: Options barrières
    products.append(('5. Down-and-Out Call', BarrierOption(S, K, T, r, sigma, 80, 'down-and-out', 'call').price()))
    products.append(('6. Down-and-In Call', BarrierOption(S, K, T, r, sigma, 80, 'down-and-in', 'call').price()))
    products.append(('7. Up-and-Out Call', BarrierOption(S, K, T, r, sigma, 120, 'up-and-out', 'call').price()))
    products.append(('8. Up-and-In Call', BarrierOption(S, K, T, r, sigma, 120, 'up-and-in', 'call').price()))

    # 9-10: Options asiatiques
    products.append(('9. Asian arithmétique', AsianOption(S, K, T, r, sigma, 'arithmetic', 'call').price()))
    products.append(('10. Asian géométrique', AsianOption(S, K, T, r, sigma, 'geometric', 'call').price()))

    # 11-12: Options lookback
    products.append(('11. Lookback floating', LookbackOption(S, K, T, r, sigma, 'floating', 'call').price()))
    products.append(('12. Lookback fixed', LookbackOption(S, K, T, r, sigma, 'fixed', 'call').price()))

    # 13: IRS
    swap = InterestRateSwap(10_000_000, 0.025, maturity=5)
    swap._generate_default_curve(0.025)
    products.append(('13. IRS (DV01)', swap.dv01()))

    # 14-15: Cap/Floor
    cap = Cap(10_000_000, 0.03, 5, 0.20)
    cap._generate_default_market()
    floor = Floor(10_000_000, 0.02, 5, 0.20)
    floor._generate_default_market()
    products.append(('14. Cap', cap.price()))
    products.append(('15. Floor', floor.price()))

    # 16-17: Repo/Forward
    products.append(('16. Repo (intérêts)', Repo(10_000_000, 0.015, 7/365).interest_amount()))
    products.append(('17. Forward (prix)', Forward(100, 0.05, 1, 0.02).forward_price()))

    print(f"\n{'Produit':<25} {'Valeur':<15}")
    print("-" * 40)
    for name, value in products:
        if abs(value) > 1000:
            print(f"{name:<25} {value:>12,.0f} €")
        else:
            print(f"{name:<25} {value:>12.4f} €")

    print(f"\nTotal: {len(products)} produits implémentés")


def main():
    """
    Exécute toutes les démonstrations.
    """
    print("\n" + "#" * 70)
    print("#" + " " * 10 + "PLATEFORME MULTI-PRODUITS DE PRICING DE DÉRIVÉS" + " " * 9 + "#")
    print("#" + " " * 20 + "Démonstration Complète" + " " * 24 + "#")
    print("#" * 70)

    demo_vanilla_options()
    demo_barrier_options()
    demo_asian_options()
    demo_lookback_options()
    demo_interest_rate_products()
    demo_repo_forward()
    demo_summary()

    print("\n" + "=" * 70)
    print("FIN DE LA DÉMONSTRATION")
    print("=" * 70)


if __name__ == "__main__":
    main()

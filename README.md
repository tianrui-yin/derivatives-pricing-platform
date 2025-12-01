# Plateforme Multi-Produits de Pricing de Dérivés

## Description

Plateforme complète de valorisation de produits dérivés, incluant options exotiques, swaps de taux, caps/floors et autres instruments.

**Niveau** : M2 Finance Quantitative - Sorbonne Université

## Fonctionnalités

- **Options Vanilles** : Call/Put européennes et américaines (arbre binomial)
- **Options Barrières** : Knock-in, Knock-out (Up/Down)
- **Options Asiatiques** : Moyenne arithmétique et géométrique
- **Options Lookback** : Sur minimum/maximum
- **Swaps de Taux (IRS)** : Valorisation et calcul du DV01
- **Caps/Floors** : Modèle de Black
- **Repos** : Accords de rachat
- **Forwards** : Contrats à terme
- **Architecture OOP** : Classes extensibles

## Structure du Projet

```
derivatives-pricing-platform/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── base_product.py       # Classe de base abstraite
│   ├── vanilla_options.py    # Options européennes et américaines
│   ├── barrier_options.py    # Options barrières
│   ├── asian_options.py      # Options asiatiques
│   ├── lookback_options.py   # Options lookback
│   ├── interest_rate.py      # IRS, Caps, Floors
│   ├── repo_forward.py       # Repos et Forwards
│   └── utils.py              # Fonctions utilitaires
├── tests/
│   └── test_pricing.py       # Tests unitaires
└── examples/
    └── demo.py               # Démonstration complète
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation Rapide

```python
from src.vanilla_options import EuropeanOption, AmericanOption
from src.barrier_options import BarrierOption
from src.asian_options import AsianOption

# Option européenne
call = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
print(f"Prix Call européen: {call.price():.4f} €")

# Option américaine (arbre binomial)
put_us = AmericanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
print(f"Prix Put américain: {put_us.price():.4f} €")

# Option barrière Down-and-Out
barrier = BarrierOption(S=100, K=100, T=1, r=0.05, sigma=0.2,
                         barrier=80, barrier_type='down-and-out')
print(f"Prix Barrière: {barrier.price():.4f} €")
```

## Produits Implémentés (15+)

1. Call européen
2. Put européen
3. Call américain
4. Put américain
5. Down-and-Out Call
6. Down-and-In Call
7. Up-and-Out Call
8. Up-and-In Call
9. Option Asiatique arithmétique
10. Option Asiatique géométrique
11. Lookback Call (sur minimum)
12. Lookback Put (sur maximum)
13. Interest Rate Swap (IRS)
14. Cap
15. Floor
16. Repo
17. Forward

## Références Théoriques

- Hull, J.C. (2018). *Options, Futures, and Other Derivatives*, 9th Edition
  - Chapitre 7 : Swaps de taux
  - Chapitre 13 : Arbres binomiaux
  - Chapitre 21 : Méthodes numériques (Monte Carlo)
  - Chapitre 26 : Options exotiques
  - Chapitre 29 : Caps, Floors, dérivés de taux

## Résultats

- 15+ produits dérivés implémentés
- Écarts < 1% vs marché

## Auteur

Projet M2 Finance Quantitative - Sorbonne Université

#  Black-Scholes Option Pricing Model CLI with Machine Learning

A command-line Python tool for pricing European call and put options using the Black-Scholes analytical formula and a machine learning regression model trained on sample market data.

---

##  Features

-  Calculate option prices using:
  -  **Black-Scholes formula**
  -  trained **machine learning regression model** (Random Forest)
-  Compare ML-predicted prices to analytical prices
-  Plot visualizations using Seaborn
-  Fully CLI-based, easy to extend and customize

---

##  Concepts Covered

- Black-Scholes option pricing theory
- Normal distribution, log-returns
- Machine learning regression (Random Forest)
- CLI app structure with `argparse`
- Data visualization with Seaborn

---

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/black-scholes-ml-cli.git
   cd black-scholes-ml-cli
---
## Commands

```python
python bs_pricer.py --mode <analytical|ml> --S <spot> --K <strike> --T <maturity> --r <rate> --sigma <volatility> --option <call|put>
```

# Plotting Example Commands
```python
python bs_pricer.py --plot price_vs_volatility
python bs_pricer.py --plot ml_vs_analytical
```

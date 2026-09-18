# Structured Products Pricing & Risk Analysis

A Python-based project for pricing a European call option and analysing
its risk characteristics using analytical and numerical methods.

## Features

- Black-Scholes analytical pricing
- Monte Carlo pricing
- Monte Carlo standard error
- 95% confidence interval
- Monte Carlo convergence analysis
- Delta, Gamma, Vega, Theta, and Rho
- Underlying-price sensitivity analysis
- Volatility sensitivity analysis
- Reproducible simulations using a fixed random seed

## Mathematical Models

The project uses the Black-Scholes model and assumes that the underlying
asset follows geometric Brownian motion under the risk-neutral measure.

The Monte Carlo method estimates the option price by simulating terminal
underlying prices and discounting the average option payoff.

## Installation

```bash
pip install -r requirements.txt
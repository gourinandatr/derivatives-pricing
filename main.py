import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Model parameters
# ============================================================

@dataclass
class OptionParameters:
    """
    Parameters for a European call option.
    """
    spot: float = 100.0
    strike: float = 100.0
    risk_free_rate: float = 0.05
    volatility: float = 0.20
    maturity: float = 1.0

    def validate(self):
        if self.spot <= 0:
            raise ValueError("Spot price must be positive.")

        if self.strike <= 0:
            raise ValueError("Strike price must be positive.")

        if self.volatility <= 0:
            raise ValueError("Volatility must be positive.")

        if self.maturity <= 0:
            raise ValueError("Maturity must be positive.")

        if not np.isfinite(self.risk_free_rate):
            raise ValueError("Risk-free rate must be finite.")


# ============================================================
# 2. Standard normal distribution functions
# ============================================================

def standard_normal_pdf(x):
    """
    Standard normal probability density function.
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def standard_normal_cdf(x):
    """
    Standard normal cumulative distribution function.

    Uses math.erf, so SciPy is not required.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ============================================================
# 3. Black-Scholes pricing
# ============================================================

def calculate_d1(params):
    """
    Calculate d1 in the Black-Scholes formula.
    """
    params.validate()

    numerator = (
        math.log(params.spot / params.strike)
        + (
            params.risk_free_rate
            + 0.5 * params.volatility ** 2
        ) * params.maturity
    )

    denominator = params.volatility * math.sqrt(params.maturity)

    return numerator / denominator


def calculate_d2(params):
    """
    Calculate d2 in the Black-Scholes formula.
    """
    d1 = calculate_d1(params)

    return d1 - params.volatility * math.sqrt(params.maturity)


def black_scholes_call_price(params):
    """
    Calculate the analytical European call price.
    """
    d1 = calculate_d1(params)
    d2 = calculate_d2(params)

    discounted_strike = (
        params.strike
        * math.exp(-params.risk_free_rate * params.maturity)
    )

    call_price = (
        params.spot * standard_normal_cdf(d1)
        - discounted_strike * standard_normal_cdf(d2)
    )

    return call_price


# ============================================================
# 4. Monte Carlo simulation
# ============================================================

def simulate_terminal_prices(params, number_of_simulations, seed=42):
    """
    Simulate terminal underlying prices using geometric
    Brownian motion under the risk-neutral measure.
    """
    params.validate()

    if number_of_simulations <= 0:
        raise ValueError("Number of simulations must be positive.")

    if seed is not None and seed < 0:
        raise ValueError("Seed must be non-negative or None.")

    rng = np.random.default_rng(seed)

    random_values = rng.standard_normal(number_of_simulations)

    drift = (
        params.risk_free_rate
        - 0.5 * params.volatility ** 2
    ) * params.maturity

    diffusion = (
        params.volatility
        * math.sqrt(params.maturity)
        * random_values
    )

    terminal_prices = params.spot * np.exp(drift + diffusion)

    return terminal_prices


def call_payoff(terminal_prices, strike):
    """
    Calculate European call payoffs.
    """
    return np.maximum(terminal_prices - strike, 0.0)


def monte_carlo_call_price(
    params,
    number_of_simulations=100_000,
    seed=42
):
    """
    Calculate the Monte Carlo call price, standard error,
    and 95% confidence interval.
    """
    terminal_prices = simulate_terminal_prices(
        params=params,
        number_of_simulations=number_of_simulations,
        seed=seed
    )

    payoffs = call_payoff(
        terminal_prices=terminal_prices,
        strike=params.strike
    )

    discounted_payoffs = (
        math.exp(-params.risk_free_rate * params.maturity)
        * payoffs
    )

    price = float(np.mean(discounted_payoffs))

    if number_of_simulations > 1:
        standard_error = float(
            np.std(discounted_payoffs, ddof=1)
            / math.sqrt(number_of_simulations)
        )
    else:
        standard_error = 0.0

    confidence_interval_lower = price - 1.96 * standard_error
    confidence_interval_upper = price + 1.96 * standard_error

    return {
        "price": price,
        "standard_error": standard_error,
        "confidence_interval": (
            confidence_interval_lower,
            confidence_interval_upper
        ),
        "terminal_prices": terminal_prices,
        "payoffs": payoffs
    }


# ============================================================
# 5. Greeks
# ============================================================

def call_delta(params):
    """
    Delta of a European call option.
    """
    d1 = calculate_d1(params)
    return standard_normal_cdf(d1)


def call_gamma(params):
    """
    Gamma of a European call option.
    """
    d1 = calculate_d1(params)

    return (
        standard_normal_pdf(d1)
        / (
            params.spot
            * params.volatility
            * math.sqrt(params.maturity)
        )
    )


def call_vega(params):
    """
    Vega of a European call option.

    This returns the price change for a volatility change
    of 1.00, not 1 percentage point.
    """
    d1 = calculate_d1(params)

    return (
        params.spot
        * standard_normal_pdf(d1)
        * math.sqrt(params.maturity)
    )


def call_theta(params):
    """
    Theta of a European call option.

    This is the derivative with respect to time to maturity.
    """
    d1 = calculate_d1(params)
    d2 = calculate_d2(params)

    first_term = -(
        params.spot
        * standard_normal_pdf(d1)
        * params.volatility
        / (2.0 * math.sqrt(params.maturity))
    )

    second_term = -(
        params.risk_free_rate
        * params.strike
        * math.exp(-params.risk_free_rate * params.maturity)
        * standard_normal_cdf(d2)
    )

    return first_term + second_term


def call_rho(params):
    """
    Rho of a European call option.

    This returns the price change for a rate change
    of 1.00, not 1 percentage point.
    """
    d2 = calculate_d2(params)

    return (
        params.strike
        * params.maturity
        * math.exp(-params.risk_free_rate * params.maturity)
        * standard_normal_cdf(d2)
    )


# ============================================================
# 6. Convergence analysis
# ============================================================

def convergence_analysis(params, simulation_counts):
    """
    Calculate Monte Carlo prices for different numbers
    of simulations.
    """
    results = []

    analytical_price = black_scholes_call_price(params)

    for count in simulation_counts:
        result = monte_carlo_call_price(
            params=params,
            number_of_simulations=count,
            seed=42
        )

        monte_carlo_price = result["price"]

        results.append({
            "simulations": count,
            "monte_carlo_price": monte_carlo_price,
            "absolute_error": abs(
                monte_carlo_price - analytical_price
            ),
            "standard_error": result["standard_error"]
        })

    return results


def plot_convergence(params, convergence_results):
    """
    Plot Monte Carlo convergence toward the
    Black-Scholes analytical price.
    """
    simulation_counts = [
        item["simulations"]
        for item in convergence_results
    ]

    monte_carlo_prices = [
        item["monte_carlo_price"]
        for item in convergence_results
    ]

    analytical_price = black_scholes_call_price(params)

    plt.figure(figsize=(9, 5))

    plt.plot(
        simulation_counts,
        monte_carlo_prices,
        marker="o",
        label="Monte Carlo price"
    )

    plt.axhline(
        y=analytical_price,
        linestyle="--",
        label="Black-Scholes price"
    )

    plt.xscale("log")
    plt.xlabel("Number of simulations")
    plt.ylabel("Option price")
    plt.title("Monte Carlo Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. Sensitivity analysis
# ============================================================

def price_for_different_spot_prices(params, spot_prices):
    """
    Calculate Black-Scholes prices for different spot prices.
    """
    prices = []

    for spot in spot_prices:
        modified_params = OptionParameters(
            spot=float(spot),
            strike=params.strike,
            risk_free_rate=params.risk_free_rate,
            volatility=params.volatility,
            maturity=params.maturity
        )

        prices.append(
            black_scholes_call_price(modified_params)
        )

    return prices


def price_for_different_volatilities(params, volatilities):
    """
    Calculate Black-Scholes prices for different volatilities.
    """
    prices = []

    for volatility in volatilities:
        modified_params = OptionParameters(
            spot=params.spot,
            strike=params.strike,
            risk_free_rate=params.risk_free_rate,
            volatility=float(volatility),
            maturity=params.maturity
        )

        prices.append(
            black_scholes_call_price(modified_params)
        )

    return prices


def plot_spot_price_sensitivity(params):
    """
    Plot option price against the underlying spot price.
    """
    spot_prices = np.linspace(60, 140, 17)
    option_prices = price_for_different_spot_prices(
        params,
        spot_prices
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        spot_prices,
        option_prices,
        marker="o"
    )

    plt.xlabel("Underlying spot price")
    plt.ylabel("European call price")
    plt.title("Sensitivity to Underlying Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_volatility_sensitivity(params):
    """
    Plot option price against volatility.
    """
    volatilities = np.linspace(0.05, 0.60, 12)
    option_prices = price_for_different_volatilities(
        params,
        volatilities
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        volatilities * 100,
        option_prices,
        marker="o"
    )

    plt.xlabel("Volatility (%)")
    plt.ylabel("European call price")
    plt.title("Sensitivity to Volatility")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. Reporting functions
# ============================================================

def print_pricing_report(params, monte_carlo_result):
    """
    Print the main pricing and risk results.
    """
    analytical_price = black_scholes_call_price(params)
    monte_carlo_price = monte_carlo_result["price"]

    confidence_lower, confidence_upper = (
        monte_carlo_result["confidence_interval"]
    )

    absolute_difference = abs(
        analytical_price - monte_carlo_price
    )

    relative_error = (
        absolute_difference / analytical_price * 100.0
    )

    print("\n" + "=" * 60)
    print("STRUCTURED PRODUCTS PRICING & RISK ANALYSIS")
    print("=" * 60)

    print("\nOPTION PARAMETERS")
    print("-" * 60)
    print(f"Spot price:          {params.spot:.4f}")
    print(f"Strike price:        {params.strike:.4f}")
    print(f"Risk-free rate:      {params.risk_free_rate:.4%}")
    print(f"Volatility:          {params.volatility:.4%}")
    print(f"Time to maturity:    {params.maturity:.4f} years")

    print("\nPRICING COMPARISON")
    print("-" * 60)
    print(f"Black-Scholes price: {analytical_price:.6f}")
    print(f"Monte Carlo price:   {monte_carlo_price:.6f}")
    print(f"Absolute difference: {absolute_difference:.6f}")
    print(f"Relative error:      {relative_error:.4f}%")

    print("\nMONTE CARLO STATISTICS")
    print("-" * 60)
    print(
        f"Standard error:      "
        f"{monte_carlo_result['standard_error']:.6f}"
    )
    print(
        f"95% confidence int.: "
        f"[{confidence_lower:.6f}, {confidence_upper:.6f}]"
    )

    print("\nGREEKS")
    print("-" * 60)
    print(f"Delta:               {call_delta(params):.6f}")
    print(f"Gamma:               {call_gamma(params):.6f}")
    print(f"Vega:                {call_vega(params):.6f}")
    print(f"Theta:               {call_theta(params):.6f}")
    print(f"Rho:                 {call_rho(params):.6f}")

    print("=" * 60)


def print_convergence_table(convergence_results):
    """
    Print convergence results in a readable table.
    """
    print("\nCONVERGENCE ANALYSIS")
    print("-" * 75)
    print(
        f"{'Simulations':>15}"
        f"{'MC Price':>15}"
        f"{'Absolute Error':>20}"
        f"{'Std. Error':>20}"
    )
    print("-" * 75)

    for item in convergence_results:
        print(
            f"{item['simulations']:>15,}"
            f"{item['monte_carlo_price']:>15.6f}"
            f"{item['absolute_error']:>20.6f}"
            f"{item['standard_error']:>20.6f}"
        )


# ============================================================
# 9. Main program
# ============================================================

def main():
    params = OptionParameters(
        spot=100.0,
        strike=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        maturity=1.0
    )

    monte_carlo_result = monte_carlo_call_price(
        params=params,
        number_of_simulations=100_000,
        seed=42
    )

    print_pricing_report(
        params=params,
        monte_carlo_result=monte_carlo_result
    )

    simulation_counts = [
        1_000,
        5_000,
        10_000,
        50_000,
        100_000,
        500_000
    ]

    convergence_results = convergence_analysis(
        params=params,
        simulation_counts=simulation_counts
    )

    print_convergence_table(convergence_results)

    plot_convergence(
        params=params,
        convergence_results=convergence_results
    )

    plot_spot_price_sensitivity(params)
    plot_volatility_sensitivity(params)


if __name__ == "__main__":
    main()
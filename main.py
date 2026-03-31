from black_scholes_strategy.model import BlackScholesModel
from black_scholes_strategy.strategy import MispricingStrategy

def main():
    # Parameters
    S = 100      # Current stock price
    K = 105      # Strike price
    T = 0.5      # Time to expiration (6 months)
    r = 0.05     # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)

    # Initialize Model
    bs_model = BlackScholesModel(S, K, T, r, sigma)

    # Calculate Theoretical Prices
    call_price = bs_model.calc_price('call')
    put_price = bs_model.calc_price('put')

    print(f"Black-Scholes Theoretical Prices:")
    print(f"  Call: {call_price:.4f}")
    print(f"  Put:  {put_price:.4f}")
    print("-" * 30)

    # Calculate Greeks
    print(f"Greeks (Call):")
    print(f"  Delta: {bs_model.calc_delta('call'):.4f}")
    print(f"  Gamma: {bs_model.calc_gamma():.4f}")
    print(f"  Vega:  {bs_model.calc_vega():.4f}")
    print(f"  Theta: {bs_model.calc_theta('call'):.4f}")
    print(f"  Rho:   {bs_model.calc_rho('call'):.4f}")
    print("-" * 30)

    # Strategy demonstration
    strategy = MispricingStrategy(threshold=0.05)

    # Scenario 1: Market price is lower than theoretical (Undervalued)
    market_call_price_1 = call_price * 0.90
    signal_1 = strategy.generate_signal(call_price, market_call_price_1)

    # Scenario 2: Market price is higher than theoretical (Overvalued)
    market_call_price_2 = call_price * 1.10
    signal_2 = strategy.generate_signal(call_price, market_call_price_2)

    print(f"Strategy Signals (Call):")
    print(f"  Scenario 1 (Market: {market_call_price_1:.4f}, Theo: {call_price:.4f}) -> {signal_1}")
    print(f"  Scenario 2 (Market: {market_call_price_2:.4f}, Theo: {call_price:.4f}) -> {signal_2}")

if __name__ == "__main__":
    main()

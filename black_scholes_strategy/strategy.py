class MispricingStrategy:
    """
    A simple strategy that identifies mispriced options by comparing
    market prices with theoretical Black-Scholes prices.
    """
    def __init__(self, threshold=0.05):
        """
        Initialize strategy.

        Parameters:
        threshold : float - Percentage threshold to trigger a signal.
                           e.g., 0.05 means market price must deviate by 5%
                           from theoretical price.
        """
        self.threshold = threshold

    def generate_signal(self, theoretical_price, market_price):
        """
        Generate a buy/sell/hold signal.

        Parameters:
        theoretical_price : float - Price calculated by Black-Scholes model.
        market_price : float - Current market price of the option.

        Returns:
        str - "BUY" if undervalued, "SELL" if overvalued, "HOLD" otherwise.
        """
        if theoretical_price <= 0:
            return "HOLD"

        diff_percent = (theoretical_price - market_price) / theoretical_price

        if diff_percent > self.threshold:
            return "BUY" # Undervalued
        elif diff_percent < -self.threshold:
            return "SELL" # Overvalued
        else:
            return "HOLD"

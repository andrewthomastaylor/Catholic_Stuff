import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    """
    Implementation of the Black-Scholes model for European options.
    """
    def __init__(self, S, K, T, r, sigma):
        """
        Initialize the model with parameters.

        Parameters:
        S : float - Current stock price
        K : float - Strike price
        T : float - Time to expiration (in years)
        r : float - Risk-free interest rate (annualized)
        sigma : float - Volatility of the underlying asset (annualized)
        """
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

    def _d1_d2(self):
        """Calculate d1 and d2 parameters."""
        if self.T <= 0:
            return 0, 0 # Handle edge case for expired options

        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def calc_price(self, option_type='call'):
        """
        Calculate the theoretical price of the option.

        Parameters:
        option_type : str - 'call' or 'put'
        """
        if self.T <= 0:
            if option_type == 'call':
                return max(0.0, self.S - self.K)
            else:
                return max(0.0, self.K - self.S)

        d1, d2 = self._d1_d2()
        if option_type == 'call':
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == 'put':
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        return price

    def calc_delta(self, option_type='call'):
        """Calculate Delta: Rate of change of price with respect to stock price."""
        if self.T <= 0:
            return 0.0 # Simplified

        d1, _ = self._d1_d2()
        if option_type == 'call':
            return norm.cdf(d1)
        elif option_type == 'put':
            return norm.cdf(d1) - 1
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def calc_gamma(self):
        """Calculate Gamma: Rate of change of Delta with respect to stock price."""
        if self.T <= 0:
            return 0.0

        d1, _ = self._d1_d2()
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        return gamma

    def calc_vega(self):
        """Calculate Vega: Rate of change of price with respect to volatility."""
        if self.T <= 0:
            return 0.0

        d1, _ = self._d1_d2()
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)
        return vega / 100.0 # Often reported per 1% change in sigma

    def calc_theta(self, option_type='call'):
        """Calculate Theta: Rate of change of price with respect to time."""
        if self.T <= 0:
            return 0.0

        d1, d2 = self._d1_d2()
        term1 = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))

        if option_type == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            theta = term1 - term2
        elif option_type == 'put':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            theta = term1 + term2
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return theta / 365.0 # Often reported per day

    def calc_rho(self, option_type='call'):
        """Calculate Rho: Rate of change of price with respect to interest rate."""
        if self.T <= 0:
            return 0.0

        _, d2 = self._d1_d2()
        if option_type == 'call':
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == 'put':
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return rho / 100.0 # Often reported per 1% change in r

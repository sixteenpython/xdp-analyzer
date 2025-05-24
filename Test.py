# Test.py

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class BlackScholesOptionPricer:
    """
    A class to calculate option prices using the Black-Scholes model.
    
    The Black-Scholes model is a mathematical model for pricing European-style options,
    based on the assumption that the price of the underlying asset follows a geometric
    Brownian motion with constant drift and volatility.
    """
    
    def __init__(self, spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Initialize the Black-Scholes option pricer with market parameters.
        
        Parameters:
        -----------
        spot_price : float
            Current price of the underlying asset
        strike_price : float
            Strike price of the option
        time_to_expiry : float
            Time to expiration in years
        risk_free_rate : float
            Risk-free interest rate (annualized)
        volatility : float
            Volatility of the underlying asset (annualized)
        dividend_yield : float, optional
            Continuous dividend yield (annualized), default is 0
        """
        self.S = spot_price
        self.K = strike_price
        self.T = time_to_expiry
        self.r = risk_free_rate
        self.sigma = volatility
        self.q = dividend_yield
        
        # Calculate d1 and d2 parameters used in the Black-Scholes formula
        self._calculate_d1_d2()
    
    def _calculate_d1_d2(self):
        """Calculate the d1 and d2 parameters for the Black-Scholes formula."""
        self.d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
    
    def call_price(self):
        """Calculate the price of a European call option."""
        return self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
    
    def put_price(self):
        """Calculate the price of a European put option."""
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
    
    def call_delta(self):
        """Calculate the delta of a call option (first derivative of option price with respect to spot price)."""
        return np.exp(-self.q * self.T) * norm.cdf(self.d1)
    
    def put_delta(self):
        """Calculate the delta of a put option."""
        return np.exp(-self.q * self.T) * (norm.cdf(self.d1) - 1)
    
    def gamma(self):
        """Calculate the gamma of an option (second derivative of option price with respect to spot price)."""
        return np.exp(-self.q * self.T) * norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        """Calculate the vega of an option (derivative of option price with respect to volatility)."""
        return self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * np.sqrt(self.T) / 100  # Divided by 100 for 1% change
    
    def call_theta(self):
        """Calculate the theta of a call option (derivative of option price with respect to time)."""
        term1 = -self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T))
        term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        term3 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)
        return (term1 + term2 + term3) / 365  # Convert to daily theta
    
    def put_theta(self):
        """Calculate the theta of a put option."""
        term1 = -self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T))
        term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        term3 = -self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
        return (term1 + term2 + term3) / 365  # Convert to daily theta
    
    def call_rho(self):
        """Calculate the rho of a call option (derivative of option price with respect to interest rate)."""
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100  # Divided by 100 for 1% change
    
    def put_rho(self):
        """Calculate the rho of a put option."""
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100  # Divided by 100 for 1% change
    
    def implied_volatility(self, option_price, option_type='call', precision=0.0001, max_iterations=100):
        """
        Calculate implied volatility using the Newton-Raphson method.
        
        Parameters:
        -----------
        option_price : float
            Market price of the option
        option_type : str, optional
            Type of option ('call' or 'put'), default is 'call'
        precision : float, optional
            Desired precision for implied volatility, default is 0.0001
        max_iterations : int, optional
            Maximum number of iterations, default is 100
            
        Returns:
        --------
        float
            Implied volatility
        """
        # Initial guess for implied volatility
        sigma = 0.2
        
        for i in range(max_iterations):
            # Save original volatility
            original_sigma = self.sigma
            
            # Set new volatility
            self.sigma = sigma
            self._calculate_d1_d2()
            
            # Calculate option price and vega
            if option_type.lower() == 'call':
                price = self.call_price()
            else:
                price = self.put_price()
            
            vega = self.vega() * 100  # Adjust vega to be for 1-point change
            
            # Calculate price difference
            price_diff = price - option_price
            
            # Check if precision is reached
            if abs(price_diff) < precision:
                # Restore original volatility
                self.sigma = original_sigma
                self._calculate_d1_d2()
                return sigma
            
            # Update sigma using Newton-Raphson
            sigma = sigma - price_diff / vega
            
            # Ensure sigma is positive
            if sigma <= 0:
                sigma = 0.001
        
        # Restore original volatility
        self.sigma = original_sigma
        self._calculate_d1_d2()
        
        raise ValueError(f"Implied volatility calculation did not converge after {max_iterations} iterations")
    
    def plot_option_prices(self, spot_range=None, volatility_range=None, time_range=None):
        """
        Plot option prices with varying parameters.
        
        Parameters:
        -----------
        spot_range : tuple, optional
            Range of spot prices (min, max, steps)
        volatility_range : tuple, optional
            Range of volatilities (min, max, steps)
        time_range : tuple, optional
            Range of times to expiry in days (min, max, steps)
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Option price vs Spot price
        if spot_range:
            min_spot, max_spot, steps = spot_range
            spot_prices = np.linspace(min_spot, max_spot, steps)
            call_prices = []
            put_prices = []
            
            original_spot = self.S
            for spot in spot_prices:
                self.S = spot
                self._calculate_d1_d2()
                call_prices.append(self.call_price())
                put_prices.append(self.put_price())
            
            # Restore original spot price
            self.S = original_spot
            self._calculate_d1_d2()
            
            ax1 = fig.add_subplot(131)
            ax1.plot(spot_prices, call_prices, 'b-', label='Call Option')
            ax1.plot(spot_prices, put_prices, 'r-', label='Put Option')
            ax1.axvline(x=self.K, color='g', linestyle='--', label='Strike Price')
            ax1.set_xlabel('Spot Price')
            ax1.set_ylabel('Option Price')
            ax1.set_title('Option Price vs Spot Price')
            ax1.legend()
            ax1.grid(True)
        
        # Plot 2: Option price vs Volatility
        if volatility_range:
            min_vol, max_vol, steps = volatility_range
            volatilities = np.linspace(min_vol, max_vol, steps)
            call_prices = []
            put_prices = []
            
            original_vol = self.sigma
            for vol in volatilities:
                self.sigma = vol
                self._calculate_d1_d2()
                call_prices.append(self.call_price())
                put_prices.append(self.put_price())
            
            # Restore original volatility
            self.sigma = original_vol
            self._calculate_d1_d2()
            
            ax2 = fig.add_subplot(132)
            ax2.plot(volatilities, call_prices, 'b-', label='Call Option')
            ax2.plot(volatilities, put_prices, 'r-', label='Put Option')
            ax2.set_xlabel('Volatility')
            ax2.set_ylabel('Option Price')
            ax2.set_title('Option Price vs Volatility')
            ax2.legend()
            ax2.grid(True)
        
        # Plot 3: Option price vs Time to Expiry
        if time_range:
            min_days, max_days, steps = time_range
            times = np.linspace(min_days, max_days, steps) / 365  # Convert days to years
            call_prices = []
            put_prices = []
            
            original_time = self.T
            for time in times:
                self.T = time
                self._calculate_d1_d2()
                call_prices.append(self.call_price())
                put_prices.append(self.put_price())
            
            # Restore original time
            self.T = original_time
            self._calculate_d1_d2()
            
            ax3 = fig.add_subplot(133)
            ax3.plot(times * 365, call_prices, 'b-', label='Call Option')
            ax3.plot(times * 365, put_prices, 'r-', label='Put Option')
            ax3.set_xlabel('Time to Expiry (days)')
            ax3.set_ylabel('Option Price')
            ax3.set_title('Option Price vs Time to Expiry')
            ax3.legend()
            ax3.grid(True)
        
        plt.tight_layout()
        plt.show()


def calculate_days_to_expiry(expiry_date):
    """
    Calculate the number of days between today and the expiry date.
    
    Parameters:
    -----------
    expiry_date : str
        Expiry date in format 'YYYY-MM-DD'
        
    Returns:
    --------
    float
        Number of years to expiry
    """
    today = datetime.now().date()
    expiry = datetime.strptime(expiry_date, '%Y-%m-%d').date()
    days = (expiry - today).days
    return max(days, 1) / 365  # Convert to years, ensure at least 1 day


def greet(name="World"):
    """Legacy greeting function kept for backward compatibility"""
    return f"Hello, {name}!"


def main():
    """Main function to demonstrate the Black-Scholes option pricing model"""
    print("Black-Scholes Option Pricing Calculator")
    print("======================================")
    
    # Example market parameters
    spot_price = 100.0
    strike_price = 100.0
    expiry_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    risk_free_rate = 0.05  # 5%
    volatility = 0.2  # 20%
    dividend_yield = 0.01  # 1%
    
    # Get user input or use defaults
    try:
        print("\nEnter option parameters (or press Enter for defaults):")
        user_spot = input(f"Spot Price [{spot_price}]: ")
        spot_price = float(user_spot) if user_spot else spot_price
        
        user_strike = input(f"Strike Price [{strike_price}]: ")
        strike_price = float(user_strike) if user_strike else strike_price
        
        user_expiry = input(f"Expiry Date (YYYY-MM-DD) [{expiry_date}]: ")
        expiry_date = user_expiry if user_expiry else expiry_date
        
        user_rate = input(f"Risk-Free Rate [{risk_free_rate}]: ")
        risk_free_rate = float(user_rate) if user_rate else risk_free_rate
        
        user_vol = input(f"Volatility [{volatility}]: ")
        volatility = float(user_vol) if user_vol else volatility
        
        user_div = input(f"Dividend Yield [{dividend_yield}]: ")
        dividend_yield = float(user_div) if user_div else dividend_yield
        
    except ValueError as e:
        print(f"Error in input: {e}")
        print("Using default values instead.")
    
    # Calculate time to expiry in years
    time_to_expiry = calculate_days_to_expiry(expiry_date)
    
    # Create Black-Scholes pricer
    pricer = BlackScholesOptionPricer(
        spot_price=spot_price,
        strike_price=strike_price,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield
    )
    
    # Calculate option prices and Greeks
    call_price = pricer.call_price()
    put_price = pricer.put_price()
    call_delta = pricer.call_delta()
    put_delta = pricer.put_delta()
    gamma = pricer.gamma()
    vega = pricer.vega()
    call_theta = pricer.call_theta()
    put_theta = pricer.put_theta()
    call_rho = pricer.call_rho()
    put_rho = pricer.put_rho()
    
    # Display results
    print("\nBlack-Scholes Option Pricing Results:")
    print("=====================================")
    print(f"Underlying Price: ${spot_price:.2f}")
    print(f"Strike Price: ${strike_price:.2f}")
    print(f"Expiry Date: {expiry_date} ({time_to_expiry*365:.0f} days)")
    print(f"Risk-Free Rate: {risk_free_rate*100:.2f}%")
    print(f"Volatility: {volatility*100:.2f}%")
    print(f"Dividend Yield: {dividend_yield*100:.2f}%")
    
    print("\nOption Prices:")
    print(f"Call Option: ${call_price:.4f}")
    print(f"Put Option: ${put_price:.4f}")
    
    print("\nOption Greeks:")
    print(f"Call Delta: {call_delta:.4f}")
    print(f"Put Delta: {put_delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Vega (per 1% change): ${vega:.4f}")
    print(f"Call Theta (per day): ${call_theta:.4f}")
    print(f"Put Theta (per day): ${put_theta:.4f}")
    print(f"Call Rho (per 1% change): ${call_rho:.4f}")
    print(f"Put Rho (per 1% change): ${put_rho:.4f}")
    
    # Ask if user wants to see option price sensitivity plots
    plot_choice = input("\nWould you like to see option price sensitivity plots? (y/n): ")
    if plot_choice.lower() == 'y':
        # Define ranges for plots
        spot_range = (spot_price * 0.7, spot_price * 1.3, 100)
        volatility_range = (0.05, 0.5, 100)
        time_range = (1, 365, 100)
        
        pricer.plot_option_prices(
            spot_range=spot_range,
            volatility_range=volatility_range,
            time_range=time_range
        )


if __name__ == "__main__":
    main()

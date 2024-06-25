import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BettingSimulation:
    """
    A class to simulate betting strategies over multiple simulations with varying probabilities and outcomes.

    Attributes:
        num_bets (int): Number of bets in each simulation.
        a (float): Lower bound for generating probabilities.
        b (float): Upper bound for generating probabilities.
        k (float): Bookmaker's margin, which affects how payouts are calculated.
        mean_val (float): Mean value used for generating the base probabilities.
        stddev_real (float): Standard deviation for the distribution around the base probabilities (real scenarios).
        stddev_bet (float): Standard deviation for modifying the base probabilities to generate bettor's perceptions.
        stddev_book (float): Standard deviation for modifying the base probabilities to generate booker's perceptions.
        fraction_kelly (float): Fraction of the Kelly criterion to use for betting (adjusts bet sizing).
        flag (boolean): Boolean to indicate whether conjugate probabilities and corresponding payout are used.
    """

    def __init__(self, num_bets, a, b, k, mean_val, stddev, betdev, bookdev, fraction_kelly=1.0, flag=True):
        self.num_bets = num_bets
        self.a = a
        self.b = b
        self.k = k
        self.mean_val = mean_val
        self.stddev = stddev
        self.betdev = betdev
        self.bookdev = bookdev
        self.fraction_kelly = fraction_kelly
        self.flag = flag

    def gen_probs(self):
        return np.clip(np.random.normal(self.mean_val, 0.1, size=self.num_bets), self.a, self.b)

    def calc_payouts(self, probs):
        impl_probs = probs / (1 - self.k)
        return 1 / impl_probs

    def mod_probs(self, base_probs, stddev):
        return np.clip(np.random.normal(base_probs, stddev), self.a, self.b)

    def kelly_criterion(self, prob, odds):
        b = odds - 1
        q = 1 - prob
        if b == 0:
            return 0
        f = (b * prob - q) / b
        return self.fraction_kelly * f if f > 0 else 0

    def simulate_bets(self, real_probs, bettor_probs, booker_probs):
        bankroll = 1
        bankroll_history = [bankroll]
        for i in range(self.num_bets):
            payout_win = self.calc_payouts(booker_probs[i])
            f_win = self.kelly_criterion(bettor_probs[i], payout_win)
            event_outcome = np.random.random() < real_probs[i]

            if self.flag:
                if f_win > 0:
                    bet_size_win = bankroll * f_win
                    bankroll += bet_size_win * (payout_win - 1) if event_outcome else -bet_size_win
            else:
                payout_lose = self.calc_payouts(1 - booker_probs[i])
                f_lose = self.kelly_criterion(1 - bettor_probs[i], payout_lose)
                if f_win > 0:
                    bet_size_win = bankroll * f_win
                    bankroll += bet_size_win * (payout_win - 1) if event_outcome else -bet_size_win
                if f_lose > 0:
                    bet_size_lose = bankroll * f_lose
                    bankroll += bet_size_lose * (payout_lose - 1) if not event_outcome else -bet_size_lose
            
            bankroll_history.append(bankroll)
        return bankroll_history
    
    def perform_simulations(self, num_simulations):
        final_bankrolls = []
        for _ in range(num_simulations):
            real_probs = self.gen_probs()
            bettor_probs = self.mod_probs(real_probs, self.betdev)
            booker_probs = self.mod_probs(real_probs, self.bookdev)

            bankroll_history = self.simulate_bets(real_probs, bettor_probs, booker_probs)
            final_bankrolls.append(bankroll_history[-1])
        
        self.plot_histogram(final_bankrolls, num_simulations)
        return final_bankrolls

    def plot_histogram(self, final_bankrolls, num_simulations):
        """
        Plots a histogram of final bankrolls to visualize the outcome distribution,
        excluding outliers based on the interquartile range (IQR).
        
        Parameters:
            final_bankrolls (list): List of final bankroll amounts from each simulation.
            num_simulations (int): The number of simulations run.
        """
        q25, q75 = np.percentile(final_bankrolls, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        filtered_bankrolls = [x for x in final_bankrolls if lower_bound <= x <= upper_bound]
        
        plt.figure(figsize=(10, 5))
        plt.hist(filtered_bankrolls, bins=250, color='blue', alpha=0.7)  
        plt.title('Histogram of Final Bankrolls after {} Simulations (Excluding Outliers)'.format(num_simulations))
        plt.xlabel('Final Bankroll ($)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def plot_bankroll_history(self, bankroll_history):
        plt.figure(figsize=(10, 5))
        plt.plot(bankroll_history, linestyle='-', linewidth=1, markersize=5)
        plt.title('Bankroll Over Time')
        plt.xlabel('Number of Bets')
        plt.ylabel('Bankroll')
        plt.grid(True)
        plt.show()

    def summarize_results(self, final_bankrolls):
        """
        Summarizes the results of the betting simulations by computing the average, min, max,
        count of bankrolls greater than 1, and an adjusted average that excludes outliers.
        
        Parameters:
            final_bankrolls (list): List of final bankroll amounts from each simulation.
        
        Returns:
            dict: Summary statistics including average, min, max, count > 1, and adjusted average.
        """
        average = np.mean(final_bankrolls)
        minimum = np.min(final_bankrolls)
        maximum = np.max(final_bankrolls)
        count_gt_one = np.sum(np.array(final_bankrolls) > 1)

        q25, q75 = np.percentile(final_bankrolls, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        adjusted_bankrolls = [x for x in final_bankrolls if lower_bound <= x <= upper_bound]
        adjusted_average = np.mean(adjusted_bankrolls) if adjusted_bankrolls else None

        return {
            'average': average,
            'minimum': minimum,
            'maximum': maximum,
            'count > 1': count_gt_one,
            'adjusted average': adjusted_average
        }

num_bets = 1230             # equivalent to the number of events per simulation
a = 0.1                     # lower bound for generating probabilities
b = 0.9                     # upper bound for generating probabilities
k = 0.1                     # bookmaker's margin
mean_val = 0.5              # mean value for generating base probabilities
stddev = 0.13               # standard deviation for generating real probabilities
betdev = 0.025              # standard deviation for bettor's probabilities (generated off real probabilities)
bookdev = 0.05              # standard deviation for bookmaker's probabilities (generated off bookmakers probabilities)
fraction_kelly = 0.8        # fraction of the Kelly criterion to be used
num_simulations = 1000      # number of simulations to run
flag = False                # flag to indicate if conjugate probabilities and corresponding payout are NOT used 

simulation = BettingSimulation(
    num_bets = num_bets,
    a = a,
    b = b,
    k = k,
    mean_val = mean_val,
    stddev = stddev,
    betdev = betdev,
    bookdev = bookdev,
    fraction_kelly = fraction_kelly,
    flag = flag
)

final_bankrolls = simulation.perform_simulations(num_simulations)
summary = simulation.summarize_results(final_bankrolls)
summary

num_bets = 1230             # equivalent to the number of events per simulation
a = 0.01                    # lower bound for generating probabilities
b = 0.09                    # upper bound for generating probabilities
k = 0.1                     # bookmaker's margin
mean_val = 0.05             # mean value for generating base probabilities
stddev = 0.013              # standard deviation for generating real probabilities
betdev = 0.005              # standard deviation for bettor's probabilities (generated off real probabilities)
bookdev = 0.010             # standard deviation for bookmaker's probabilities (generated off bookmakers probabilities)
fraction_kelly = 0.3        # fraction of the Kelly criterion to be used
num_simulations = 1000      # number of simulations to run
flag = True                 # flag to indicate whether conjugate probabilities and corresponding payout are used

simulation = BettingSimulation(
    num_bets = num_bets,
    a = a,
    b = b,
    k = k,
    mean_val = mean_val,
    stddev = stddev,
    betdev = betdev,
    bookdev = bookdev,
    fraction_kelly = fraction_kelly,
    flag = flag
)

final_bankrolls = simulation.perform_simulations(num_simulations)
summary = simulation.summarize_results(final_bankrolls)
summary



def run_kelly_variation_simulation():
    num_bets = 1230
    a = 0.1
    b = 0.9
    k = 0.1
    mean_val = 0.5
    stddev = 0.13
    betdev = 0.025
    bookdev = 0.05
    num_simulations = 1000
    flag = False

    kelly_values = np.arange(0.1, 0.82, 0.02)  
    adjusted_averages = []

    for fraction_kelly in kelly_values:
        simulation = BettingSimulation(
            num_bets=num_bets,
            a=a,
            b=b,
            k=k,
            mean_val=mean_val,
            stddev=stddev,
            betdev=betdev,
            bookdev=bookdev,
            fraction_kelly=fraction_kelly,
            flag=flag
        )
        final_bankrolls = simulation.perform_simulations(num_simulations)
        summary = simulation.summarize_results(final_bankrolls)
        adjusted_averages.append(summary.get('adjusted average', 0))  

    plt.figure(figsize=(10, 5))
    plt.plot(kelly_values, adjusted_averages, marker='o', linestyle='-', color='blue')
    plt.title('Adjusted Average Bankroll by Fraction Kelly')
    plt.xlabel('Fraction Kelly')
    plt.ylabel('Adjusted Average Bankroll')
    plt.grid(True)
    plt.show()

run_kelly_variation_simulation()

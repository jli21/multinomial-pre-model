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
        Parameters:
            final_bankrolls (list): List of final bankroll amounts from each simulation.
        
        Returns:
            dict: Summary statistics...
        """
        
        q25, q75 = np.percentile(final_bankrolls, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        adjusted_bankrolls = [x for x in final_bankrolls if lower_bound <= x <= upper_bound]

        average = np.mean(final_bankrolls)
        median = np.median(final_bankrolls)
        minimum = np.min(final_bankrolls)
        maximum = np.max(final_bankrolls)
        variance = np.var(adjusted_bankrolls)
        std_dev = np.std(adjusted_bankrolls)
        
        bkt_lt_0_5 = np.mean(np.array(final_bankrolls) < 0.5)
        bkt_0_5_to_1 = np.mean((np.array(final_bankrolls) >= 0.5) & (np.array(final_bankrolls) < 1))
        bkt_1_to_2 = np.mean((np.array(final_bankrolls) >= 1) & (np.array(final_bankrolls) < 2))
        bkt_2_to_4 = np.mean((np.array(final_bankrolls) >= 2) & (np.array(final_bankrolls) < 4))
        bkt_gt_4 = np.mean(np.array(final_bankrolls) >= 4)
        bkt_gt_1 = np.mean(np.array(final_bankrolls) > 1)  

        adjusted_average = np.mean(adjusted_bankrolls) if adjusted_bankrolls else None

        return {
            'average': average,
            'median': median,
            'minimum': minimum,
            'maximum': maximum,
            'variance': variance,
            'standard deviation': std_dev,
            'bucket (p < 0.5)': bkt_lt_0_5,
            'bucket (0.5 < p < 1)': bkt_0_5_to_1,
            'bucket (1 < p < 2)': bkt_1_to_2,
            'bucket (2 < p < 4)': bkt_2_to_4,
            'bucket (p > 4)': bkt_gt_4,
            'bucket (p > 1)': bkt_gt_1,
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
fraction_kelly = 0.3        # fraction of the Kelly criterion to be used
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
    res = {}

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
        res[f"{fraction_kelly:.2f}"] = summary

    plot_summary_statistics(res)
    return res

def plot_summary_statistics(res):
    plt.figure(figsize=(20, 32))  
    num_plots = len(next(iter(res.values())))
    cols = 2  
    rows = num_plots // cols + (num_plots % cols > 0)
    kelly_values = [float(k) for k in res.keys()]

    for i, key in enumerate(next(iter(res.values())).keys(), start=1):
        values = [res[k][key] for k in res]
        ax = plt.subplot(rows, cols, i)
        ax.plot(kelly_values, values, linestyle='-', color='blue', linewidth=2)  
        ax.set_title(f'{key} by Fraction Kelly', fontsize=14)  
        ax.set_xlabel('Fraction Kelly', fontsize=12)  
        ax.set_ylabel(key, fontsize=12)  
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=10)  

    plt.tight_layout()  
    plt.show()

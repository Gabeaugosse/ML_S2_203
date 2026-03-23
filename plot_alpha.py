import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import default_params

from classes.game import Game
from strategies import *

def main():
    num_turns = 50000

    alphas = [0.001, 0.01, 0.1, 0.5, 0.9, 1]
    colors = ['#E63946', '#F4A261', '#E9C46A', '#2A9D8F', '#264653', '#000000']

    fig, ax1 = plt.subplots(figsize=(10, 6))

    for alpha, color in zip(alphas, colors):
        print(f"\nRunning alpha={alpha}")
        ql_params = {
            "alpha"         : alpha,    
            "gamma"         : 0.9,    
            "epsilon"       : 1.0,     
            "epsilon_min"   : 0.01,    
            "epsilon_decay" : 0.9999, # Fast decay to isolate alpha's effect on convergence
        }
        
        strategy_mix = {
            TwoTitsForTat: 0.5,
            QLearningStrategy: 0.5
        }
        
        game = Game(num_players=2, num_turns=num_turns, strategy_mix=strategy_mix, ql_params=ql_params)
        game.play()
        
        ql_agent = [a for a in game.players.values() if isinstance(a.get_strategy(), QLearningStrategy)][0]
        hist = ql_agent.get_strategy().history
        
        T = len(hist["action"])
        window = max(200, T // 100)
        
        actions_bin = np.array([1 if a == "C" else 0 for a in hist["action"]], dtype=float)
        coop_rate = np.convolve(actions_bin, np.ones(window) / window, mode="valid")
        turns_roll = np.arange(window, T + 1)
        
        ax1.plot(turns_roll, coop_rate, label=f"α = {alpha}", color=color, lw=1.5)

    ax1.set_title("Impact of Learning Rate (α) on convergence against TwoTitsForTat (50K turns)")
    ax1.set_xlabel("Turn")
    ax1.set_ylabel(f"Cooperation rate (window={window})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    output_path = "alpha_impact_ttft.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()

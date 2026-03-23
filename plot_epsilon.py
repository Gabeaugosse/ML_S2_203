import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import default_params

from classes.game import Game
from strategies import TitForTat, QLearningStrategy

def main():
    num_turns = default_params.NUM_TURNS

    decays = [0.999, 0.9999, 0.99999]
    colors = ['#E63946', '#2A9D8F', '#E9C46A']

    fig, ax1 = plt.subplots(figsize=(10, 6))

    for decay, color in zip(decays, colors):
        print(f"\nRunning decay={decay}")
        ql_params = {
            "alpha"         : 0.1,    
            "gamma"         : 0.9,    
            "epsilon"       : 1.0,     
            "epsilon_min"   : 0.01,    
            "epsilon_decay" : decay, 
        }
        
        strategy_mix = {
            TitForTat: 0.5,
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
        
        ax1.plot(turns_roll, coop_rate, label=f"decay = {decay}", color=color, lw=1.5)

    ax1.set_title("Impact of Exploration Decay Rate (ε) on Cooperation against TitForTat")
    ax1.set_xlabel("Turn")
    ax1.set_ylabel(f"Cooperation rate (window={window})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    output_path = "epsilon_decay.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()

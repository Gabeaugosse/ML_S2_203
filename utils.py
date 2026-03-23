import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from strategies import QLearningStrategy
import matplotlib.cm as cm


def plot_convergence(game, num_turns: int, output_path: str = "convergence.png") -> None:
    """
    Generate and save a figure illustrating the convergence of the
    Q-learning policy, plotting the cooperation rate over time.
    """
    # --- Retrieve all Q-learning agents from the population ---
    ql_agents = [
        agent for agent in game.players.values()
        if isinstance(agent.get_strategy(), QLearningStrategy)
    ]
    if not ql_agents:
        print("No Q-learning agent found; figure not generated.")
        return

    # Select the first Q-learning agent for general parameters
    strat_0 = ql_agents[0].get_strategy()
    hist_0  = strat_0.history
    T     = len(hist_0["action"])   # Total no. of steps recorded
    turns = np.arange(1, T + 1)
    window = max(200, T // 100)

    # ===================== FIGURE LAYOUT =====================
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Convergence of the Q-agent policy\n"
        f"(α={strat_0.alpha}, γ={strat_0.gamma}, "
        f"ε_min={strat_0.epsilon_min}, decay={strat_0.epsilon_decay})",
        fontsize=13, fontweight="bold"
    )

  
    colors = cm.tab10(np.linspace(0, 1, max(10, len(ql_agents))))

    for i, agent in enumerate(ql_agents):
        strat = agent.get_strategy()
        hist  = strat.history
        
        actions_bin = np.array([1 if a == "C" else 0 for a in hist["action"]], dtype=float)
        if len(actions_bin) >= window:
            coop_rate = np.convolve(actions_bin, np.ones(window) / window, mode="valid")
            turns_roll  = np.arange(window, len(actions_bin) + 1)
            
            label = f"Agent {i+1}" if len(ql_agents) > 1 else "Cooperation rate"
            c = colors[i % 10] if len(ql_agents) > 1 else "#2A9D8F"
            ax1.plot(turns_roll, coop_rate, color=c, lw=1.5, alpha=0.8, label=label)

    ax1.set_ylabel("Cooperation rate")
    ax1.tick_params(axis="y")
    ax1.set_ylim(-0.05, 1.15)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.set_xlabel("Step")
    if len(ql_agents) <= 10:
        ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(f"Agents behaviour over time (window={window})")
    ax1.grid(True, alpha=0.3)

    # Secondary y-axis: ε schedule
    color_eps = "#000000"
    ax1b = ax1.twinx()
    ax1b.plot(turns, hist_0["epsilon_hist"], color=color_eps, lw=1, alpha=0.7, label="ε (exploration rate)")
    ax1b.set_ylabel("ε", color=color_eps)
    ax1b.tick_params(axis="y", labelcolor=color_eps)
    ax1b.set_ylim(-0.05, 1.15)
    ax1b.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {os.path.abspath(output_path)}")
    plt.close()
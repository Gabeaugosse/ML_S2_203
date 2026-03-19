import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no Tcl/Tk required) — must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from strategies import QLearningStrategy


def plot_convergence(game, num_turns: int, output_path: str = "convergence.png") -> None:
    """
    Generate and save a two-panel figure illustrating the convergence of the
    Q-learning agent toward the systematic defection policy.
    """
    # --- Retrieve all Q-learning agents from the population ---
    ql_agents = [
        agent for agent in game.players.values()
        if isinstance(agent.get_strategy(), QLearningStrategy)
    ]
    if not ql_agents:
        print("No Q-learning agent found; figure not generated.")
        return

    # Select the first Q-learning agent for analysis
    strat = ql_agents[0].get_strategy()
    hist  = strat.history
    T     = len(hist["action"])   # Total no. of steps recorded
    turns = np.arange(1, T + 1)

    # --- Compute rolling defection rate (uniform moving avg. over 'window' steps) ---
    window = max(200, T // 100)
    actions_bin = np.array([1 if a == "B" else 0 for a in hist["action"]], dtype=float)
    betray_rate = np.convolve(actions_bin, np.ones(window) / window, mode="valid")
    turns_roll  = turns[window - 1:]

    # --- Theoretical fixed-point Q* values (Bellman optimality, infinite horizon) ---
    # Q*(B) = r_B / (1 - γ)  with r_B = 4  →  Q*(B) = 4 / (1 - γ)
    # Q*(C) = r_C + γ · Q*(B)  with r_C = 3  →  Q*(C) = 3 + γ · Q*(B)
    gamma         = strat.gamma
    q_star_betray = 4 / (1 - gamma)          # Optimal Q-val for defection
    q_star_coop   = 3 + gamma * q_star_betray # Optimal Q-val for cooperation

    # ===================== FIGURE LAYOUT =====================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        "Convergence of the Q-agent toward systematic defection\n"
        f"(α={strat.alpha}, γ={strat.gamma}, "
        f"ε_min={strat.epsilon_min}, decay={strat.epsilon_decay})",
        fontsize=13, fontweight="bold"
    )

    # ---- Panel 1: Rolling defection rate + exploration rate (ε) ----
    color_betray = "#E63946"
    color_eps    = "#457B9D"

    ax1.plot(turns_roll, betray_rate, color=color_betray, lw=1.5,
             label=f"Defection rate (window={window})")
    ax1.axhline(1.0, color=color_betray, lw=0.8, ls="--", alpha=0.5, label="Optimal policy (100% D)")
    ax1.set_ylabel("Defection rate", color=color_betray)
    ax1.tick_params(axis="y", labelcolor=color_betray)
    ax1.set_ylim(-0.05, 1.15)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Agent behaviour over time")
    ax1.grid(True, alpha=0.3)

    # Secondary y-axis: ε schedule
    ax1b = ax1.twinx()
    ax1b.plot(turns, hist["epsilon_hist"], color=color_eps, lw=1, alpha=0.7, label="ε (exploration rate)")
    ax1b.set_ylabel("ε", color=color_eps)
    ax1b.tick_params(axis="y", labelcolor=color_eps)
    ax1b.set_ylim(-0.05, 1.15)
    ax1b.legend(loc="upper right", fontsize=9)

    # ---- Panel 2: Q-value trajectories vs. theoretical optima ----
    color_qb = "#E63946"
    color_qc = "#2A9D8F"

    ax2.plot(turns, hist["q_betray"], color=color_qb, lw=1.5, label="Q((C,C), Defect)")
    ax2.plot(turns, hist["q_coop"],   color=color_qc, lw=1.5, label="Q((C,C), Cooperate)")
    ax2.axhline(q_star_betray, color=color_qb, lw=0.8, ls="--", alpha=0.6,
                label=f"Q*(Defect) = {q_star_betray:.1f}")
    ax2.axhline(q_star_coop,   color=color_qc, lw=0.8, ls="--", alpha=0.6,
                label=f"Q*(Cooperate) = {q_star_coop:.1f}")
    ax2.set_ylabel("Q-value")
    ax2.set_xlabel("Step")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_title("Q-value convergence for state (C, C)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {os.path.abspath(output_path)}")
    plt.close()
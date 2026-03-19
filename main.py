from classes.game import Game
import default_params
import argparse
from strategies import *
import matplotlib
matplotlib.use('Agg')  # Backend sans fenêtre (pas besoin de Tcl/Tk)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

def plot_convergence(game: Game, num_turns: int, output_path: str = "convergence.png") -> None:
    """
    Génère et sauvegarde un graphe combiné montrant la convergence du Q-agent
    vers la trahison systématique.
    """
    # --- Récupérer le(s) Q-agent(s) ---
    ql_agents = [
        agent for agent in game.players.values()
        if isinstance(agent.get_strategy(), QLearningStrategy)
    ]
    if not ql_agents:
        print("Aucun Q-agent trouvé, graphe non généré.")
        return

    # On prend le premier Q-agent
    strat = ql_agents[0].get_strategy()
    hist  = strat.history
    T     = len(hist["action"])
    turns = np.arange(1, T + 1)

    # --- Rolling betrayal rate ---
    window = max(200, T // 100)
    actions_bin = np.array([1 if a == "B" else 0 for a in hist["action"]], dtype=float)
    betray_rate = np.convolve(actions_bin, np.ones(window) / window, mode="valid")
    turns_roll  = turns[window - 1:]

    # --- Valeurs théoriques ---
    gamma = strat.gamma
    q_star_betray = 4 / (1 - gamma)   # 40
    q_star_coop   = 3 / (1 - gamma) * gamma + 3  # === 3 + gamma*q_star_betray if we take max=B
    # Re-compute properly: Q*(B)=40, Q*(C)=3+0.9*40=39
    q_star_betray = 4 / (1 - gamma)
    q_star_coop   = 3 + gamma * q_star_betray

    # ===================== FIGURE =====================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Convergence du Q-agent vers la trahison systématique\n"
                 f"(α={strat.alpha}, γ={strat.gamma}, "
                 f"ε_min={strat.epsilon_min}, decay={strat.epsilon_decay})",
                 fontsize=13, fontweight="bold")

    # ---- Panneau 1 : Taux de trahison + Epsilon ----
    color_betray = "#E63946"
    color_eps    = "#457B9D"

    ax1.plot(turns_roll, betray_rate, color=color_betray, lw=1.5,
             label=f"Taux de trahison (fenêtre={window})")
    ax1.axhline(1.0, color=color_betray, lw=0.8, ls="--", alpha=0.5, label="Optimal (100% B)")
    ax1.set_ylabel("Taux de trahison", color=color_betray)
    ax1.tick_params(axis="y", labelcolor=color_betray)
    ax1.set_ylim(-0.05, 1.15)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Comportement de l'agent")
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(turns, hist["epsilon_hist"], color=color_eps, lw=1, alpha=0.7, label="ε (epsilon)")
    ax1b.set_ylabel("Epsilon", color=color_eps)
    ax1b.tick_params(axis="y", labelcolor=color_eps)
    ax1b.set_ylim(-0.05, 1.15)
    ax1b.legend(loc="upper right", fontsize=9)

    # ---- Panneau 2 : Q-valeurs ----
    color_qb = "#E63946"
    color_qc = "#2A9D8F"

    ax2.plot(turns, hist["q_betray"], color=color_qb, lw=1.5, label="Q((C,C), Trahir)")
    ax2.plot(turns, hist["q_coop"],   color=color_qc, lw=1.5, label="Q((C,C), Coopérer)")
    ax2.axhline(q_star_betray, color=color_qb, lw=0.8, ls="--", alpha=0.6,
                label=f"Q*(Trahir) = {q_star_betray:.1f}")
    ax2.axhline(q_star_coop,   color=color_qc, lw=0.8, ls="--", alpha=0.6,
                label=f"Q*(Coopérer) = {q_star_coop:.1f}")
    ax2.set_ylabel("Valeur Q")
    ax2.set_xlabel("Tour")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_title("Convergence des Q-valeurs (état = (C,C))")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Graphe sauvegardé : {os.path.abspath(output_path)}")
    plt.close()


def main() :

    # Easier to execute the code with custom params from the terminal
    parser = argparse.ArgumentParser(description="Simulation params")

    parser.add_argument(
        "--nb_players",
        type=int,
        default=None,
        help="Number of players (default: from default_params)"
    )
    parser.add_argument(
        "--nb_turns",
        type=int,
        default=None,
        help="Number of turns (default: from default_params)"
    )

    args = parser.parse_args()

    num_players = args.nb_players if args.nb_players is not None else default_params.NUM_PLAYERS
    num_turns   = args.nb_turns   if args.nb_turns   is not None else default_params.NUM_TURNS

    strategy_mix = {
    TitForTat:          0,
    AlwaysCooperate:    0.5,
    AlwaysBetray:       0,
    RandomAction:       0,
    Joss:               0,
    Bully:              0,
    TitForTwoTats:      0,
    QLearningStrategy: 0.5
}
    # Q-Learning hyperparameters
    ql_params = {
        "alpha"         : 0.1,    # Learning rate
        "gamma"         : 0.9,    # Discount factor
        "epsilon"       : 1.0,    # Initial exploration rate
        "epsilon_min"   : 0.01,   # Minimum exploration rate
        "epsilon_decay" : 0.9999, # Multiplicative decay per step
    }

    # Play the game
    print(f"\nSimulation with {num_players} players who will play against each other {num_turns} times !\n")
    game = Game(num_players=num_players, num_turns=num_turns, strategy_mix=strategy_mix, ql_params=ql_params)
    game.play()
    plot_convergence(game, num_turns)
    



if __name__ == "__main__" :
    main()
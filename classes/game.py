import default_params
from classes.agent import Agent
from strategies import *
import utils
from itertools import combinations
import random
import math

class Game() :
    def __init__(self, num_players: int, num_turns: int, strategy_mix: dict):
        """ strategy_mix : dict mapping Strategy -> proportion (float), must sum to 1.
        """
        self.num_players = num_players
        self.num_turns = num_turns

        self.logs = [] # History of all interactions among the game
        self.players = self._init_players(strategy_mix)
        self.players_indexes = list(self.players.keys())


    def _init_players(self, strategy_mix: dict) -> dict:
            """
            Instanciate agents depending on input proportion in the main file.
            """
            assert abs(sum(strategy_mix.values()) - 1.0) < 1e-6, "Proportion must sum to 1 !!"

            # Calcul des effectifs bruts
            raw_counts = {s: p * self.num_players for s, p in strategy_mix.items()}
            floor_counts = {s: math.floor(c) for s, c in raw_counts.items()}

            # Distribution du reste
            remainder = self.num_players - sum(floor_counts.values())
            sorted_by_remainder = sorted(strategy_mix.keys(),
                                        key=lambda s: raw_counts[s] - floor_counts[s],
                                        reverse=True)
            for i in range(remainder):
                floor_counts[sorted_by_remainder[i]] += 1

            # Instanciation des agents
            players = {}
            idx = 0
            for strategy, count in floor_counts.items():
                for _ in range(count):
                    players[idx] = Agent(idx, strategy)
                    idx += 1

            print("Start population :")
            for s, c in floor_counts.items():
                print(f"  {s.__name__:25s} : {c} agent(s) ({c/self.num_players:.1%})")
            print("\n")
            return players


    def play(self) -> None:
        """ List every possible combination of players, and make them play NUM_TURNS times."""
        pairs = list(combinations(self.players_indexes, 2))
        random.shuffle(pairs)
        for p1, p2 in pairs:
            print(p1,p2)
            for _ in range(self.num_turns) :
                self.play_match(p1,p2)
        
        self.print_metrics()

    
    def play_match(self, p1_id: int, p2_id: int) -> None :
        """ Solve an interaction between two players."""
        # Instanciate each player into a variable
        player_1 = self.players[p1_id] 
        player_2 = self.players[p2_id]
        # print(f"Player 1 score: {player_1.get_score()}, Player 2 score: {player_2.get_score()}")

        # Give the opponnent id to each player for them to choose their action
        first_player_action = player_1.choose_action(p2_id)
        second_player_action = player_2.choose_action(p1_id)
        # print(first_player_action, second_player_action)

        # Compute the outcome
        gains = default_params.GAIN_MATRIX[default_params.ACTIONS_INDEX[first_player_action]][default_params.ACTIONS_INDEX[second_player_action]]
        
        player_1_gain = gains[0]
        player_2_gain = gains[1]

        # Update scores
        player_1.update_score(player_1_gain)
        player_2.update_score(player_2_gain)
        
        # print(player_1_gain,player_2_gain)
        
        # Update the logs and players history
        player_1.update_interactions(p2_id, {"player_action" : first_player_action, 
                                            "opponent_action" : second_player_action})
        player_2.update_interactions(p1_id, {"player_action" : second_player_action, 
                                            "opponent_action" : first_player_action})
            

        
    def compute_metrics(self) -> dict:
        """
        Calcule l'ensemble des métriques de fin de partie.
        """
        stats = {}

        for pid, agent in self.players.items():
            total_interactions = sum(len(v) for v in agent.interactions.items())
            
            all_my_actions    = [r["player_action"]   for hist in agent.interactions.values() for r in hist]
            all_opp_actions   = [r["opponent_action"]  for hist in agent.interactions.values() for r in hist]

            n = len(all_my_actions)

            coop_rate_self = all_my_actions.count("C")  / n if n > 0 else 0
            coop_rate_opp  = all_opp_actions.count("C") / n if n > 0 else 0

            # Taux d'exploitation : j'ai trahi pendant que l'autre coopérait
            exploit_rate = sum(
                1 for a, b in zip(all_my_actions, all_opp_actions) if a == "B" and b == "C"
            ) / n if n > 0 else 0

            # Taux de victimisation : l'autre a trahi pendant que je coopérais
            victimized_rate = sum(
                1 for a, b in zip(all_my_actions, all_opp_actions) if a == "C" and b == "B"
            ) / n if n > 0 else 0

            # Taux de coopération mutuelle (outcome CC)
            mutual_coop_rate = sum(
                1 for a, b in zip(all_my_actions, all_opp_actions) if a == "C" and b == "C"
            ) / n if n > 0 else 0

            # Taux de trahison mutuelle (outcome BB)
            mutual_betray_rate = sum(
                1 for a, b in zip(all_my_actions, all_opp_actions) if a == "B" and b == "B"
            ) / n if n > 0 else 0

            # Score moyen par interaction
            avg_score = agent.score / n if n > 0 else 0

            stats[pid] = {
                "strategy"          : str(agent.strategy),
                "total_score"       : agent.score,
                "avg_score"         : round(avg_score, 4),
                "n_interactions"    : n,
                "coop_rate_self"    : round(coop_rate_self, 4),
                "coop_rate_opp"     : round(coop_rate_opp, 4),
                "mutual_coop_rate"  : round(mutual_coop_rate, 4),
                "mutual_betray_rate": round(mutual_betray_rate, 4),
                "exploit_rate"      : round(exploit_rate, 4),
                "victimized_rate"   : round(victimized_rate, 4),
            }

        # --- Classement global ---
        ranked = sorted(stats.items(), key=lambda x: x[1]["total_score"], reverse=True)
        winner_id, winner_stats = ranked[0]

        # --- Agrégation par stratégie ---
        strategy_groups = {}
        for pid, s in stats.items():
            name = s["strategy"]
            if name not in strategy_groups:
                strategy_groups[name] = []
            strategy_groups[name].append(s)

        strategy_summary = {}
        for name, group in strategy_groups.items():
            strategy_summary[name] = {
                "n_agents"              : len(group),
                "total_score"          : sum(g["total_score"] for g in group),
                "avg_score_per_agent"  : round(sum(g["total_score"] for g in group) / len(group), 4),
                "avg_coop_rate"        : round(sum(g["coop_rate_self"] for g in group) / len(group), 4),
                "avg_exploit_rate"     : round(sum(g["exploit_rate"] for g in group) / len(group), 4),
                "avg_victimized_rate"  : round(sum(g["victimized_rate"] for g in group) / len(group), 4),
                "avg_mutual_coop_rate" : round(sum(g["mutual_coop_rate"] for g in group) / len(group), 4),
            }

        best_strategy = max(strategy_summary.items(), key=lambda x: x[1]["avg_score_per_agent"])

        return {
            "winner"            : {"id": winner_id, **winner_stats},
            "ranking"           : [(pid, s["strategy"], s["total_score"]) for pid, s in ranked],
            "per_agent"         : stats,
            "per_strategy"      : strategy_summary,
            "best_strategy"     : {"name": best_strategy[0], **best_strategy[1]},
        }
    
    def print_metrics(self) -> None:
        m = self.compute_metrics()

        print("\n" + "="*55)
        print("  RÉSULTATS DE LA SIMULATION")
        print("="*55)

        print(f"\n🏆 Gagnant individuel : Agent {m['winner']['id']}")
        print(f"   Stratégie  : {m['winner']['strategy']}")
        print(f"   Score total: {m['winner']['total_score']}")
        print(f"   Score moyen: {m['winner']['avg_score']} / interaction")

        print(f"\n🥇 Meilleure stratégie : {m['best_strategy']['name']}")
        print(f"   Score moyen / agent : {m['best_strategy']['avg_score_per_agent']}")
        print(f"   Taux coopération    : {m['best_strategy']['avg_coop_rate']:.1%}")
        print(f"   Taux exploitation   : {m['best_strategy']['avg_exploit_rate']:.1%}")

        print("\n📊 Classement par stratégie :")
        ranked_strats = sorted(m["per_strategy"].items(),
                            key=lambda x: x[1]["avg_score_per_agent"], reverse=True)
        print(f"  {'Stratégie':<25} {'Agents':>6} {'Score moy':>10} {'Coop':>7} {'Exploit':>8} {'Victimisé':>10} {'CC':>7}")
        print("  " + "-"*75)
        for name, s in ranked_strats:
            print(f"  {name:<25} {s['n_agents']:>6} {s['avg_score_per_agent']:>10.2f} "
                f"{s['avg_coop_rate']:>7.1%} {s['avg_exploit_rate']:>8.1%} "
                f"{s['avg_victimized_rate']:>10.1%} {s['avg_mutual_coop_rate']:>7.1%}")
        print("="*55)




    def random_matching(self) -> list :
        """ Takes every player initial id and shuffle it to do a perfect match shuffled list

        Returns:
            list: Shuffled list
        """
        return utils.shuffle(self.players_indexes.copy())

from abc import ABC, abstractmethod
import random
import default_params

class Strategy(ABC):
    """ Common interface for predefined strategies. (Like interface in Java)
    """

    @abstractmethod
    def choose_action(self, my_id: str, other_player_id: str, interactions: dict) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class AlwaysCooperate(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """No matter what, cooperate.
        """
        return "C"

    def __str__(self) -> str:
        return "AlwaysCooperate"


class AlwaysBetray(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """No matter what, betrays.
        """
        return "B"

    def __str__(self) -> str:
        return "AlwaysBetray"


class RandomAction(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Takes a random choice at each interaction.
        """
        return random.choice(["C", "B"])

    def __str__(self) -> str:
        return "RandomAction"


class ProbaCooperation(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Cooperate with a fixed probability given in the default params.
        """
        return random.choices(["C", "B"], weights=[default_params.P_COOP, 1 - default_params.P_COOP])[0]

    def __str__(self) -> str:
        return f"ProbaCooperation(p={default_params.P_COOP})"

class TitForTat(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Cooperates on the first round, then imitates opponent's previous move.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) == 0:
            return "C"
        return interactions[other_player_id][-1]["opponent_action"]

    def __str__(self) -> str:
        return "TitForTat"

class SuspiciousTitForTat(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Defects on the first round, then imitates opponent's previous move.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) == 0:
            return "B"
        return interactions[other_player_id][-1]["opponent_action"]

    def __str__(self) -> str:
        return "SuspiciousTitForTat"

class TitForTwoTats(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Defects only if opponent defected twice in a row.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) < 2:
            return "C"
        last_two = [r["opponent_action"] for r in interactions[other_player_id][-2:]]
        return "B" if last_two == ["B", "B"] else "C"

    def __str__(self) -> str:
        return "TitForTwoTats"

class TwoTitsForTat(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Defects twice after being defected against, otherwise cooperates.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) == 0:
            return "C"
        last_two = [r["opponent_action"] for r in interactions[other_player_id][-2:]]
        if "B" in last_two:
            return "B"
        return "C"

    def __str__(self) -> str:
        return "TwoTitsForTat"

class DiscriminatingAltruist(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """In the Optional IPD, cooperates with any player that has never defected against it,
        otherwise refuses to engage.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) == 0:
            return "C"
        ever_defected = any(r["opponent_action"] == "B" for r in interactions[other_player_id])
        return "B" if ever_defected else "C"

    def __str__(self) -> str:
        return "DiscriminatingAltruist"
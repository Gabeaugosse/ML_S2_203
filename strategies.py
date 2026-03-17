from abc import ABC, abstractmethod
import random
import default_params

class Strategy(ABC) :
    """ Common interface for predefined strategies. (Like interface in Java)
    """

    @abstractmethod
    def choose_action(self, my_id : str, other_player_id : str, interactions : dict) -> str:
        pass


class AlwaysCooperate(Strategy) :
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """No matter what, cooperate.
        """
        return "C"

class AlwaysBetray(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """No matter what, betrays.
        """
        return "B"

class RandomAction(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Takes a random choice at each interaction.
        """
        return random.choice(["C", "B"])

class ProbaCooperation(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Cooperate with a fixed probability given in the default params.
        """
        return random.choices(["C", "D"], weights=[default_params.P_COOP, 1-default_params.P_COOP])[0]


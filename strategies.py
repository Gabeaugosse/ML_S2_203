from abc import ABC, abstractmethod


class Strategy(ABC) :
    """ Common interface for predefined strategies. (Like interface in Java)
    """

    @abstractmethod
    def choose_action(self, my_id : str, other_player_id : str, interactions : dict) -> str:
        pass


class AlwaysCooperate(Strategy) :
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        print(interactions)
        return "C"

class AlwaysBetray(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        print(interactions)
        return "B"

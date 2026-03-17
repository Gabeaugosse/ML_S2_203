import default_params
from classes.agent import Agent
from strategies import *
import utils

class Game() :
    def __init__(self, num_players, num_turns):
        self.num_players = num_players
        self.num_turns = num_turns

        self.logs = [] # History of all interactions among the game
        self.players = {i : Agent(i,AlwaysCooperate) for i in range(self.num_players)} # All agent in the game
        self.players_indexes = [i for i in range(self.num_players)]
    

    def play(self) -> None :
        for _ in range(self.num_turns) :
            self.turn()
    
    def turn(self) -> None :
        matching = self.random_matching() # At each turn, we first do a random shuffling to match every agent with another one
        
        # Iterate through each players meetings
        for p1, p2 in zip(self.players_indexes, matching) :
            # print(f"Player {p1} will be facing player {p2}")
            
            # Instanciate each player into a variable
            player_1 = self.players[p1] 
            player_2 = self.players[p2]

            # Give the opponnent id to each player for them to choose their action
            first_player_action = player_1.choose_action(p2)
            second_player_action = player_2.choose_action(p1)
            # print(first_player_action, second_player_action)

            # Compute the outcome
            gains = default_params.GAIN_MATRIX[default_params.ACTIONS_INDEX[first_player_action]][default_params.ACTIONS_INDEX[second_player_action]]
            # print(gains)
            player_1_gain = gains[0]
            player_2_gain = gains[1]

            # print(player_1_gain,player_2_gain)
            
            # Update the logs and players history
            player_1.update_interactions(p2, (first_player_action, second_player_action))
            player_2.update_interactions(p1, (second_player_action, first_player_action))

        

    def random_matching(self) -> list :
        """ Takes every player initial id and shuffle it to do a perfect match shuffled list

        Returns:
            list: Shuffled list
        """
        return utils.shuffle(self.players_indexes.copy())

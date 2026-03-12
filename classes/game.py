import params

class Game() :
    def __init__(self):
        self.num_players = params.NUM_PLAYERS
        self.num_turns = params.NUM_TURNS

        self.logs = [] # History of all interactions among the game
        self.players = [] # All agent in the game (try even number first)
    
    def create_players(self) -> None :
        pass
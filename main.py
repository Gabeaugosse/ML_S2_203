from classes.game import Game
import default_params
import argparse

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


    # Play the game
    game = Game(num_players=num_players, num_turns=num_turns)
    game.play()
    



if __name__ == "__main__" :
    main()
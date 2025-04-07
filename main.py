from uttt import UTTT, InvalidMoveException
from mcts import predict as mcts_predict
from minimax import predict as minimax_predict
from dql import predict as dql_predict
import time

def print_game_state(game):
    print("\n    0       1       2")
    
    for i in range(3):
        for j in range(3):
            print(f"{i if j == 1 else ' '} ", end="")
            
            for k in range(3):
                sb = game.subboards[i][k]
                row = []
                for l in range(3):
                    val = sb._query((j, l))
                    if val is None:
                        if game.next_subboard is not None and game.next_subboard == (i, k):
                            row.append("·")
                        else:
                            row.append("-")
                    elif val == 1:
                        row.append("x")
                    else:
                        row.append("o")
                
                print("|".join(row), end="")
                if k < 2:
                    print(" | ", end="")
            
            print()
        
        if i < 2:
            print("  " + "-" * 21)

def play_game(x: str, o: str, vis=False, final_stats=False):
    x_name = x
    if x == "Minimax":
        x_predict = minimax_predict
    elif x == "MCTS":
        x_predict = mcts_predict
    elif x == "DQL":
        x_predict = dql_predict
    else:
        return
    x_timer = 0

    o_name = o
    if o == "Minimax":
        o_predict = minimax_predict
    elif o == "MCTS":
        o_predict = mcts_predict
    elif o == "DQL":
        o_predict = dql_predict
    else:
        return
    o_timer = 0
    
    game = UTTT()
    move_count = 0
    
    if vis:
        print(f"Starting game: {x_name} (X) vs {o_name} (O)")
        print_game_state(game)
    
    while game.game_winner() is None:
        current_agent_name = x_name if game.current_player == 'X' else o_name
        predict_function = x_predict if game.current_player == 'X' else o_predict
        
        try:
            start = time.time()
            player = game.current_player
            move = predict_function(game)
            board_pos, move_pos = move
            game = game.make_move(board_pos, move_pos)
            move_count += 1
            move_duration = time.time() - start
            if player == 'X':
                x_timer += move_duration
            else:
                o_timer += move_duration
            
            if vis:
                print(f"\nMove {move_count}: {player} played at board {board_pos}, position {move_pos}")
                print(f"   *** {current_agent_name} took {move_duration:.3f}s")
                print_game_state(game)
                input("")
            
        except InvalidMoveException as e:
            print(f"Game ended due to invalid move by {current_agent_name}: {e}")
            return
        except Exception as e:
            print(f"Game ended due to error from {current_agent_name}: {e}")
            return
    
    winner = game.game_winner()
    winner_name = x_name if winner == 'X' else o_name if winner == 'O' else "Tie"
    
    if winner == 'T':
        print("Game over!\t --> tie")
    else:
        print(f"Game over!\t --> Winner: {winner_name} ({winner})")

    if final_stats:
        print(f"Moves made: {move_count}\n")

        print("\nFinal board:")
        print_game_state(game)        

        print("\nCumulative move timers:")
        print(f"   {x_name} (X):\t{x_timer:.3f}s\tavg: {x_timer/move_count:.3f}s/move")
        print(f"   {o_name} (O):\t{o_timer:.3f}s\tavg: {o_timer/move_count:.3f}s/move\n")

minimax = "Minimax"
mcts = "MCTS"
dql = "DQL"
# for i in range(10):
#     print(f"*** Game {i + 1} ***")
#     play_game(x=mcts, o=minimax)
play_game(x=mcts, o=minimax, final_stats=True)

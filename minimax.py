from uttt import UTTT

MAX_DEPTH = 6 # 5 -> .16 for a move , 6 -> .25s , 7 -> 1.45s

# utility evaluation weights
GAME_WIN_UTILITY = 1000
SUBBOARD_WIN_UTILITY = 10
CENTRAL_POSITION_UTILITY = 2

# reward when a game ends
def end_reward(winner, player):
    if winner == "T": return 0
    if winner == player: return GAME_WIN_UTILITY
    return -GAME_WIN_UTILITY

# active game utility function based on: (1) winning subboards, (2) claiming subboard centeral positions
def evaluate_game(state: UTTT, player: str):
    score = 0
    opponent = 'O' if player == 'X' else 'X'

    for subboard in [state.subboards[i][j] for i in range(3) for j in range(3)]:
        if subboard.winner == player:
            score += SUBBOARD_WIN_UTILITY
        elif subboard.winner == opponent:
            score -= SUBBOARD_WIN_UTILITY
        elif subboard.winner is None:
            center = subboard._query((1, 1))
            if center is not None:
                if (center == 1 and player == 'X') or (center == 0 and player == 'O'):
                    score += CENTRAL_POSITION_UTILITY
                else:
                    score -= CENTRAL_POSITION_UTILITY

    return score

# returns the best (score, move) for the player
def minimax(state: UTTT, depth: int, alpha: float, beta: float, maximizing: bool, player: str):
    winner = state.game_winner()
    
    # base case: evaluate leaf nodes
    if winner or depth == 0:
        if winner:
            return end_reward(winner, player), None
        return evaluate_game(state, player), None
    
    # branching
    valid_moves = state.get_valid_moves()
    if not valid_moves:
        return 0, None
    
    # find the min/max move
    best_move = None
    
    if maximizing:
        best_score = float('-inf')
        for move in valid_moves:
            new_state = state.make_move(*move)
            score, _ = minimax(new_state, depth - 1, alpha, beta, False, player)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
            # prune search space
            if beta <= alpha:
                break
    else:
        best_score = float('inf')
        for move in valid_moves:
            new_state = state.make_move(*move)
            score, _ = minimax(new_state, depth - 1, alpha, beta, True, player)
            
            if score < best_score:
                best_score = score
                best_move = move
            
            beta = min(beta, best_score)
            # prune search space
            if beta <= alpha:
                break
    
    return best_score, best_move

# predict players best move in the given game state; minimax from this state as the root node, using alpha-beta pruning
def predict(state: UTTT, player: str):
    _, best_move = minimax(state, MAX_DEPTH, float('-inf'), float('inf'), True, player)
    return best_move

print(predict(UTTT(), 'X'))

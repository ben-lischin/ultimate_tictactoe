from uttt import UTTT

MAX_DEPTH = 5

# evaluation weights
GAME_WIN_UTILITY = 1000 # highest priority: win the game
SUBBOARD_WIN_UTILITY = 50 # win a subboard
CENTRAL_POSITION_UTILITY = 2 # claim a subboard central position
CLOSE_TO_SUBBOARD_WIN_UTILITY = 5 # setting up a win on your next move in a subboard (subject to opponent's move)
GIVE_OP_SUBBOARD_PENALTY = 10 # when you force your opponent to play in a subboard they will win in a single move

# evaluate a terminal game state: win/loss/tie
def end_reward(winner, player):
    if winner == "T": return 0
    if winner == player: return GAME_WIN_UTILITY
    return -GAME_WIN_UTILITY

# if a sequence of 3 slots has 1 empty and 2 bits for the given player
def is_winnable_sequence(line, subboard, player_bit):
    count = sum(1 for pos in line if subboard._query(pos) == player_bit)
    empties = sum(1 for pos in line if subboard._query(pos) is None)
    return count == 2 and empties == 1

# if the given player can win a subboard on their next move
def is_winnable_subboard(subboard, player_bit):
    # rows/cols
    for i in range(3):
        row = [(i, j) for j in range(3)]
        col = [(j, i) for j in range(3)]
        if is_winnable_sequence(row, subboard, player_bit) or is_winnable_sequence(col, subboard, player_bit):
            return True

    # diagonals
    diagonals = [
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)]
    ]
    for diag in diagonals:
        if is_winnable_sequence(diag, subboard, player_bit):
            return True

    return False

# active game utility function based on: (1) winning subboards, (2) claiming subboard centeral positions, (3) setting up subboards to win in a single move, (4) sending opponent into a subboard they will win in 1 move
def evaluate_game(state: UTTT, player: str):
    score = 0
    opponent = 'O' if player == 'X' else 'X'
    player_bit = 1 if player == 'X' else 0
    opponent_bit = 1 - player_bit

    for i in range(3):
        for j in range(3):
            subboard = state.subboards[i][j]

            # subboard wins
            if subboard.winner == player:
                score += SUBBOARD_WIN_UTILITY
            elif subboard.winner == opponent:
                score -= SUBBOARD_WIN_UTILITY

            elif subboard.winner is None:
                # valuable center position
                center = subboard._query((1, 1))
                if center == player_bit:
                    score += CENTRAL_POSITION_UTILITY
                elif center == opponent_bit:
                    score -= CENTRAL_POSITION_UTILITY

                 # 1 move away from winning a subboard
                if is_winnable_subboard(subboard, player_bit):
                    score += CLOSE_TO_SUBBOARD_WIN_UTILITY
                if is_winnable_subboard(subboard, opponent_bit):
                    score -= CLOSE_TO_SUBBOARD_WIN_UTILITY

    # consider the next subboard we are forcing the opponent to play in
    if state.next_subboard is not None:
        next_board = state.subboards[state.next_subboard[0]][state.next_subboard[1]]
        if next_board.winner is None:
            # penalize if sending opponent to a subboard they will win in a single move
            if is_winnable_subboard(next_board, opponent_bit):
                score -= GIVE_OP_SUBBOARD_PENALTY
            if is_winnable_subboard(next_board, player_bit):
                score += GIVE_OP_SUBBOARD_PENALTY

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
def predict(state: UTTT):
    _, best_move = minimax(state, MAX_DEPTH, float('-inf'), float('inf'), True, state.current_player)
    return best_move

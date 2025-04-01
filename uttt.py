from copy import deepcopy
from subboard import SubBoard

class InvalidMoveException(Exception): pass


class UTTT:
    """
    Ultimate Tic-Tac-Toe game model

    ...
    """

    def __init__(self):
        # 3x3 matrix of SubBoards
        self.subboards = [[SubBoard() for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        # subboard next move must be played in
        self.next_subboard = None
        # None (in progress) | 'X' | 'O' | 'T' (tie)
        self.winner = None

    def make_move(self, board: tuple, move: tuple):
        # check valid player moving and move validity at highest level
        if not self._is_valid_board(board) or not self._satisfies_next_subboard(board):
            raise InvalidMoveException(f" Player {self.current_player} made an invalid move")
        
        # check move validity in lower level
        if not self._get_subboard(board).valid_move(move):
            raise InvalidMoveException(f" Player {self.current_player} made an invalid move")
        
        new_state = deepcopy(self)
        new_state._get_subboard(board).make_move(self.current_player, move)

        # check for win
        new_state._update_winner(board)
        
        new_state.current_player = new_state._next_player()
        new_state.next_subboard = new_state._next_subboard(move)
        
        return new_state

    def game_winner(self): return self.winner

    def _is_valid_board(self, board: tuple):
        return 0 <= board[0] < 3 and 0 <= board[1] < 3
    
    def _satisfies_next_subboard(self, board: tuple):
        return self.next_subboard is None or board == self.next_subboard
    
    def _update_winner(self, board: tuple):
        possible_winner = self.subboards[board[0]][board[1]].winner
        if possible_winner is None:
            return
        
        # check row win || col win || either diagonal win
        if (
            all(x.winner == possible_winner for x in self.subboards[board[0]]) or 
            all(x.winner == possible_winner for x in [self.subboards[i][board[1]] for i in range(3)]) or 
            possible_winner == self.subboards[0][0].winner == self.subboards[1][1].winner == self.subboards[2][2].winner or
            possible_winner == self.subboards[0][2].winner == self.subboards[1][1].winner == self.subboards[2][0].winner
        ):
            self.winner = possible_winner
       
        # full board = tie
        elif all(all(x.winner is not None for x in row) for row in self.subboards):
            self.winner = 'T'

    def _next_player(self):
        return 'O' if self.current_player == 'X' else 'X'
    
    def _next_subboard(self, board: tuple):
        # if the next board is complete, player can move anywhere
        if self.subboards[board[0]][board[1]].winner:
            return None
        
        return board

    # outputs list of valid move tuples: ((board_x, board_y),(move_x, move_y))[]
    def get_valid_moves(self):
        valid_moves = []

        if self.next_subboard is not None:
            self._get_valid_moves_subboard(*self.next_subboard, valid_moves)
        else:
            for board_row in range(3):
                for board_col in range(3):
                    self._get_valid_moves_subboard(board_row, board_col, valid_moves)

        return valid_moves

    def _get_valid_moves_subboard(self, board_row: int, board_col: int, valid_moves: list):
        subboard = self.subboards[board_row][board_col]
        if subboard.winner:
            return

        for subboard_row in range(3):
            for subboard_col in range(3):
                if subboard.board[subboard_row][subboard_col] is None:
                    valid_moves.append(((board_row, board_col), (subboard_row, subboard_col)))
        return
    
    def _get_subboard(self, board): return self.subboards[board[0]][board[1]]

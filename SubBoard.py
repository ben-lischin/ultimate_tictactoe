class SubBoard:
    """
    Model for a sub-board in the larger Ultimate Tic-Tac-Toe game

    ...
    """

    def __init__(self):
        self.board = [[None for _ in range(3)] for _ in range(3)]
        # None (in progress) | 'X' | 'O' | 'T' (tie)
        self.winner = None

    def make_move(self, player: str, move: tuple):
        if not self._is_valid_move(move):
            return False
        
        self.board[move[0]][move[1]] = player

        self._update_winner(move)

        return True

    def _is_valid_move(self, move: tuple):
        return self.winner is None and self._is_on_board(move) and self.board[move[0]][move[1]] is None

    def _is_on_board(self, move: tuple):
        return 0 <= move[0] < 3 and 0 <= move[1] < 3

    def _update_winner(self, move: tuple):
        player = self.board[move[0]][move[1]]

        # check row win || col win || either diagonal win
        if (
            all(x == player for x in self.board[move[0]]) or
            all(x == player for x in [self.board[i][move[1]] for i in range(3)]) or
            player == self.board[0][0] == self.board[1][1] == self.board[2][2] or
            player == self.board[0][2] == self.board[1][1] == self.board[2][0]
        ):
            self.winner = player
       
        # full board = tie
        elif all(all(x is not None for x in row) for row in self.board):
            self.winner = 'T'

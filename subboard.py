class SubBoard:
    """
    Model for a sub-board in the larger Ultimate Tic-Tac-Toe game

    Occupied bit: 1
    X bit: 1
    """

    def __init__(self):
        self.occupied = 0
        self.board = 0
        self.winner = None

    def make_move(self, player: str, move: tuple):
        if not self.valid_move(move):
            return False

        player_bit = 1 if player=="X" else 0
        self._place(player_bit, move)
        self._update_winner(move)

        return True

    def _place(self, val, pos: tuple):
        ind = 3 * pos[0] + pos[1]
        self.board |= (val << ind)
        self.occupied |= (1 << ind)

    def _query(self, pos: tuple):
        ind = 3 * pos[0] + pos[1]
        if not (self.occupied >> ind) & 1: return None
        return (self.board >> ind) & 1

    def get_valid_moves(self):
        if self.winner is not None: return []

        open_bit_inds = []
        for i in range(9): 
            if (self.occupied >> i) & 1 == 0:
                open_bit_inds.append(i)
        return [((i // 3), (i % 3)) for i in open_bit_inds]

    def valid_move(self, move: tuple):
        return self.winner is None and self._is_on_board(move) and self._query(move) is None

    def _is_on_board(self, move: tuple):
        return 0 <= move[0] < 3 and 0 <= move[1] < 3

    def _update_winner(self, move: tuple):
        player_bit = self._query(move)

        # check row win || col win || either diagonal win
        if (
            all(self._query((move[0],i)) == player_bit for i in range(3)) or
            all(self._query((i,move[1])) == player_bit for i in range(3)) or
            player_bit == self._query((0,0)) == self._query((1,1)) == self._query((2,2)) or
            player_bit == self._query((0,2)) == self._query((1,1)) == self._query((2,0))
        ):
            self.winner = "X" if player_bit == 1 else "O"
       
        # full board = tie
        elif self.occupied == (1<<9)-1:
            self.winner = 'T'

    def __repr__(self) -> str:
        def pos_to_str(pos):
            res = self._query(pos)
            if res == None: return "_"
            if res == 1: return "X"
            if res == 0: return "O"

        out = "\n"
        for r in range(3):
            slots = [pos_to_str((r, c)) for c in range(3)]
            out += f"{slots[0]}|{slots[1]}|{slots[2]}\n"
        return out

    def copy(self):
        out = SubBoard()
        out.occupied = self.occupied
        out.board = self.board
        out.winner = self.winner
        return out

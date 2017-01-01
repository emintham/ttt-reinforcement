import random


class TTT:
    def __init__(self):
        self.last_move = None
        self.board = [0] * 9
        self.current_player = 1

    def __str__(self):
        def str_pos(i, x):
            if x > 0:
                return '*'
            elif x < 0:
                return 'X'
            else:
                return str(i)

        def str_row(i):
            row = [(3*i, self.board[3*i]),
                   (3*i + 1, self.board[3*i + 1]),
                   (3*i + 2, self.board[3*i + 2])]
            return '|' + '|'.join(str_pos(j, x) for j, x in row) + '|'

        output = ''
        output += '-------\n'
        output += '\n'.join(str_row(i) for i in range(3)) + '\n'
        output += '-------\n'

        return output

    @property
    def moves(self):
        return [i
                for i, x in enumerate(self.board)
                if x == 0]

    @property
    def winner(self):
        winning_lines = [(0, 1, 2),
                         (3, 4, 5),
                         (6, 7, 8),
                         (0, 3, 6),
                         (1, 4, 7),
                         (2, 5, 8),
                         (0, 4, 8),
                         (2, 4, 6)]
        for a, b, c in winning_lines:
            if self.board[a] != 0 and self.board[a] == self.board[b] and self.board[b] == self.board[c]:
                return self.board[a]

    @property
    def draw(self):
        return len(self.moves) == 0

    def play(self, pos):
        self.board[pos] = self.current_player
        self.last_move = pos
        self.current_player *= -1

    def undo(self):
        if self.last_move is None:
            raise Exception

        self.board[self.last_move] = 0
        self.last_move = None
        self.current_player *= -1

    def try_play(self, pos):
        self.play(pos)
        feature_vec = self.as_feature_vector()
        self.undo()

        return feature_vec

    def as_feature_vector(self):
        return tuple(self.board)

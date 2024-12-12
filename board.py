class TicTacToeBoard:

    def __init__(self):
        self.dim_s = 9
        self.dim_a = 9
        self.reset()

    def reset(self):
        self.player = 1
        self.board = [0 for s in range(self.dim_s)]

    def get_state(self):
        return tuple(self.board)

    def get_valid_actions(self):
        return [a for a, s in enumerate(self.board) if s == 0]

    def win_or_draw(self):
        row1 = [self.board[0], self.board[1], self.board[2]]
        row2 = [self.board[3], self.board[4], self.board[5]]
        row3 = [self.board[6], self.board[7], self.board[8]]
        col1 = [self.board[0], self.board[3], self.board[6]]
        col2 = [self.board[1], self.board[4], self.board[7]]
        col3 = [self.board[2], self.board[5], self.board[8]]
        dia1 = [self.board[0], self.board[4], self.board[8]]
        dia2 = [self.board[2], self.board[4], self.board[6]]
        lines = [row1, row2, row3, col1, col2, col3, dia1, dia2]
        if any(set(line) == {1} for line in lines):
            return 1
        if all(s != 0 for s in self.board):
            return 0
        return None

    def step(self, action):
        self.board[action] = 1
        wod = self.win_or_draw()
        if wod == 1:
            return self.get_state(), 100, True
        if wod == 0:
            return self.get_state(), 0, True
        return self.get_state(), 0, False

    def exchange_player(self):
        self.board = [-s for s in self.board]
        self.player = -self.player

    def render(self):
        for i in range(3):
            for j in range(3):
                chess = self.board[i * 3 + j]
                if chess == self.player:
                    print("X", end=" ")
                elif chess == -self.player:
                    print("O", end=" ")
                else:
                    print("-", end=" ")
            print()
        print()

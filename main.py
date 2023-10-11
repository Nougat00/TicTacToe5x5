from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax

class TicTacToe4x4(TwoPlayerGame):
    def __init__(self, players):
        self.players = players
        self.board = [[0 for _ in range(5)] for _ in range(5)]
        self.current_player = 1

    def possible_moves(self):
        moves = []
        for i in range(5):
            for j in range(5):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def make_move(self, move):
        i, j = move
        self.board[i][j] = self.current_player

    def lose(self):
        # Sprawdzamy, czy któryś z graczy wygrał w poziomie, pionie lub na przekątnych
        for i in range(4):
            # Sprawdzanie wierszy
            if self.board[i][0] == self.board[i][1] == self.board[i][2] == self.board[i][3] == self.board[i][4] != 0:
                return True
            # Sprawdzanie kolumn
            if self.board[0][i] == self.board[1][i] == self.board[2][i] == self.board[3][i] == self.board[i][4] != 0:
                return True

        # Sprawdzanie przekątnych
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == self.board[3][3] == self.board[4][4] != 0:
            return True

        if self.board[0][4] == self.board[1][3] == self.board[2][2] == self.board[3][1] == self.board[4][4] != 0:
            return True

        return False

    def is_over(self):
        return self.lose() or len(self.possible_moves()) == 0

    def show(self):
        for i in range(5):
            for j in range(5):
                if self.board[i][j] == 1:
                    print("X", end=" ")
                elif self.board[i][j] == 2:
                    print("O", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()

    def scoring(self):
        return 100 if game.lose() else 0

if __name__ == "__main__":
    ai_algo = Negamax(6)
    human = Human_Player()
    game = TicTacToe4x4([human, AI_Player(ai_algo)])
    game.play()

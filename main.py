from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax


"""This is extended TicTacToe Game that has 5x5 board.
We can use command 'show moves' to display available moves.
To move we should choose one move from list by typing 'move #number' ex 'move #1'
To win player has to has 5 marks in row, column or diagonally.

Created by: Jakub Gola & Bartosz Laskowski
"""
class TicTacToe5x5(TwoPlayerGame):
    def __init__(self, players):
        """Initializes the Tic-Tac-Toe game with a 5x5 board and the specified players.

            Parameters:
            players (list): Players List (np. [Human_Player(), AI_Player()]).
        """
        self.players = players
        self.board = [[0 for _ in range(5)] for _ in range(5)]
        self.current_player = 1

    def possible_moves(self):
        # Returns a list of available moves as tuples (row, column).
        moves = []
        for i in range(5):
            for j in range(5):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def make_move(self, move):
        """Makes a move on the board based on the specified move (row, column).

        Parameters:
        move (tuple): Ruch w formie krotki (row, column).
        """
        i, j = move
        self.board[i][j] = self.current_player

    def lose(self):
        """Checks if the current player has lost the game.

        Returns:
        bool: True if player lost, otherwise False
        """
        # We check whether one of the players won horizontally, vertically or diagonally
        for i in range(4):
            # Checking rows
            if self.board[i][0] == self.board[i][1] == self.board[i][2] == self.board[i][3] == self.board[i][4] != 0:
                return True
            # Checking columns
            if self.board[0][i] == self.board[1][i] == self.board[2][i] == self.board[3][i] == self.board[i][4] != 0:
                return True

        # Checking diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == self.board[3][3] == self.board[4][4] != 0:
            return True

        if self.board[0][4] == self.board[1][3] == self.board[2][2] == self.board[3][1] == self.board[4][4] != 0:
            return True

        return False

    def is_over(self):
        """Checks if the game is over, either due to a player winning or no available moves.

        Returns:
        bool: True if game is over, otherwise False
        """
        return self.lose() or len(self.possible_moves()) == 0

    def show(self):
        # Displays the current state of the game on the console.
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
        """Calculates the game score.

        Returns:
            int: Returns -10 if the current player has lost, else returns 10.
        """
        return -10 if game.lose() else 10


if __name__ == "__main__":
    ai_algo = Negamax(6)
    human = Human_Player()
    game = TicTacToe5x5([human, AI_Player(ai_algo)])
    game.play()

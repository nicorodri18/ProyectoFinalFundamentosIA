import math
import numpy as np
from connect4.policy import Policy


class Nicrack(Policy):

    def mount(self, *args):
        pass

    def moves(self, board):

        valid = []

        for c in range(7):

            if board[0][c] == 0:
                valid.append(c)

        return valid

    def play(self, board, col, piece):

        temp = board.copy()

        for r in range(5, -1, -1):

            if temp[r][col] == 0:
                temp[r][col] = piece
                break

        return temp

    def winner(self, board, piece):

        # Horizontal

        for r in range(6):
            for c in range(4):

                if (
                    board[r][c] == piece and
                    board[r][c + 1] == piece and
                    board[r][c + 2] == piece and
                    board[r][c + 3] == piece
                ):
                    return True

        # Vertical

        for r in range(3):
            for c in range(7):

                if (
                    board[r][c] == piece and
                    board[r + 1][c] == piece and
                    board[r + 2][c] == piece and
                    board[r + 3][c] == piece
                ):
                    return True

        # Diagonal positiva

        for r in range(3):
            for c in range(4):

                if (
                    board[r][c] == piece and
                    board[r + 1][c + 1] == piece and
                    board[r + 2][c + 2] == piece and
                    board[r + 3][c + 3] == piece
                ):
                    return True

        # Diagonal negativa

        for r in range(3, 6):
            for c in range(4):

                if (
                    board[r][c] == piece and
                    board[r - 1][c + 1] == piece and
                    board[r - 2][c + 2] == piece and
                    board[r - 3][c + 3] == piece
                ):
                    return True

        return False

    # Heurística sencilla:
    # dar bonus al centro

    def evaluate(self, col):

        if col == 3:
            return 0.2

        return 0

    def simulate(self, board, piece):

        current = piece

        temp = board.copy()

        while True:

            possible = self.moves(temp)

            if len(possible) == 0:
                return 0

            # Prioridad al centro

            possible.sort(
                key=lambda x: abs(3 - x)
            )

            move = int(
                np.random.choice(possible[:3])
            )

            temp = self.play(
                temp,
                move,
                current
            )

            if self.winner(temp, current):
                return current

            current *= -1

    def act(self, s):

        simulations = getattr(self, "simulations", 35)

        possible = self.moves(s)

        # Prioridad al centro

        possible.sort(
            key=lambda x: abs(3 - x)
        )

        red = np.count_nonzero(s == -1)
        yellow = np.count_nonzero(s == 1)

        if red <= yellow:
            piece = -1
        else:
            piece = 1

        rival = -piece

        # Jugada ganadora inmediata

        for col in possible:

            temp = self.play(
                s,
                col,
                piece
            )

            if self.winner(temp, piece):
                return col

        # Bloquear rival

        for col in possible:

            temp = self.play(
                s,
                col,
                rival
            )

            if self.winner(temp, rival):
                return col

        wins = {}
        visits = {}

        for move in possible:

            wins[move] = 1
            visits[move] = 1

        total = len(possible)

        for _ in range(simulations):

            best = possible[0]
            best_ucb = -999999

            for move in possible:

                value = (
                    wins[move] /
                    visits[move]
                )

                exploration = math.sqrt(
                    math.log(total) /
                    visits[move]
                )

                ucb = (
                    value +
                    1.4 * exploration
                )

                if ucb > best_ucb:

                    best_ucb = ucb
                    best = move

            temp = self.play(
                s,
                best,
                piece
            )

            result = self.simulate(
                temp,
                rival
            )

            visits[best] += 1
            total += 1

            bonus = self.evaluate(best)

            if result == piece:

                wins[best] += 1 + bonus

            elif result == 0:

                wins[best] += 0.5 + bonus

        best_move = possible[0]
        best_score = -1

        for move in possible:

            score = (
                wins[move] /
                visits[move]
            )

            if score > best_score:

                best_score = score
                best_move = move

        return best_move
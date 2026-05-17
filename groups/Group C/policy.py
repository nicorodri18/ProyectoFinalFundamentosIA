import numpy as np
from connect4.policy import Policy


class OhYes(Policy):

    def mount(self, *args):
        pass

    def act(self, s):

        posibles = []

        for i in range(7):
            if s[0][i] == 0:
                posibles.append(i)

        rojas = np.count_nonzero(s == -1)
        amarillas = np.count_nonzero(s == 1)

        if rojas <= amarillas:
            ficha = -1
        else:
            ficha = 1

        rival = -ficha

        for col in posibles:

            tablero = s.copy()

            for fila in range(5, -1, -1):
                if tablero[fila][col] == 0:
                    tablero[fila][col] = ficha
                    break

            for f in range(6):
                for c in range(4):

                    if (
                        tablero[f][c] == ficha and
                        tablero[f][c + 1] == ficha and
                        tablero[f][c + 2] == ficha and
                        tablero[f][c + 3] == ficha
                    ):
                        return col

            for f in range(3):
                for c in range(7):

                    if (
                        tablero[f][c] == ficha and
                        tablero[f + 1][c] == ficha and
                        tablero[f + 2][c] == ficha and
                        tablero[f + 3][c] == ficha
                    ):
                        return col

        for col in posibles:

            tablero = s.copy()

            for fila in range(5, -1, -1):
                if tablero[fila][col] == 0:
                    tablero[fila][col] = rival
                    break

            for f in range(6):
                for c in range(4):

                    if (
                        tablero[f][c] == rival and
                        tablero[f][c + 1] == rival and
                        tablero[f][c + 2] == rival and
                        tablero[f][c + 3] == rival
                    ):
                        return col

            for f in range(3):
                for c in range(7):

                    if (
                        tablero[f][c] == rival and
                        tablero[f + 1][c] == rival and
                        tablero[f + 2][c] == rival and
                        tablero[f + 3][c] == rival
                    ):
                        return col

        if 3 in posibles:
            return 3

        return int(np.random.choice(posibles))
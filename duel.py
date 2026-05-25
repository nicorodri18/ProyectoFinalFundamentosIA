import importlib.util

from connect4.connect_state import ConnectState

# -------- BASIC --------

spec_basic = importlib.util.spec_from_file_location(
    "basic",
    "groups/Group C/backup_basic.py"
)

module_basic = importlib.util.module_from_spec(spec_basic)

spec_basic.loader.exec_module(module_basic)

BasicAgent = module_basic.Nicrack

# -------- PRO --------

spec_pro = importlib.util.spec_from_file_location(
    "pro",
    "groups/Group C/policy.py"
)

module_pro = importlib.util.module_from_spec(spec_pro)

spec_pro.loader.exec_module(module_pro)

ProAgent = module_pro.Nicrack

# -------- RESULTADOS --------

basic_wins = 0
pro_wins = 0
draws = 0

# -------- 10 PARTIDAS --------

for game in range(10):

    basic = BasicAgent()

    pro = ProAgent()

    state = ConnectState()

    turn = -1

    while True:

        board = state.board

        if turn == -1:

            move = basic.act(board)

        else:

            move = pro.act(board)

        state = state.transition(move)

        if state.is_final():

            winner = state.get_winner()

            if winner == -1:

                basic_wins += 1

            elif winner == 1:

                pro_wins += 1

            else:

                draws += 1

            print(f"Partida {game + 1}: ganador -> {winner}")

            break

        turn *= -1

# -------- RESULTADOS FINALES --------

print("\nRESULTADOS FINALES")

print("Basic wins:", basic_wins)

print("Pro wins:", pro_wins)

print("Empates:", draws)
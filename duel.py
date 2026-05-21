import importlib.util

from connect4.connect_state import ConnectState

# -------- BASIC --------

spec_basic = importlib.util.spec_from_file_location(
    "basic",
    "groups/Group C/policy_basic_backup.py"
)

module_basic = importlib.util.module_from_spec(spec_basic)

spec_basic.loader.exec_module(module_basic)

BasicAgent = module_basic.OhYes

# -------- PRO --------

spec_pro = importlib.util.spec_from_file_location(
    "pro",
    "groups/Group C/policy.py"
)

module_pro = importlib.util.module_from_spec(spec_pro)

spec_pro.loader.exec_module(module_pro)

ProAgent = module_pro.OhYes

# -------- INSTANCIAS --------

basic = BasicAgent()
pro = ProAgent()

# -------- ESTADO --------

state = ConnectState()

turn = -1

# -------- PARTIDA --------

while True:

    board = state.board

    if turn == -1:

        move = basic.act(board)

        print("\nBASIC juega:", move)

    else:

        move = pro.act(board)

        print("\nPRO juega:", move)

    state = state.transition(move)

    print(state.board)

    if state.is_final():

        winner = state.get_winner()

        print("\nGanador:", winner)

        break

    turn *= -1
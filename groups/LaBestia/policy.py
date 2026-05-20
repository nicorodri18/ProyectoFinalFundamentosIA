import numpy as np
import math
from typing import Optional
from connect4.policy import Policy
from connect4.connect_state import ConnectState

class MCTSNode:
    __slots__ = ("state", "parent", "action", "children",
                 "value", "visits", "untried")

    def __init__(self, state: ConnectState,
                 parent: Optional["MCTSNode"] = None,
                 action: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: list["MCTSNode"] = []
        self.value = 0.0
        self.visits = 0
        self.untried: list[int] = (
            list(state.get_free_cols()) if not state.is_final() else []
        )

    def ucb1(self, c: float) -> float:
        """UCB1: explota (value/visits) + explora (sqrt(log N / n))"""
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, c: float) -> "MCTSNode":
        return max(self.children, key=lambda ch: ch.ucb1(c))

    def expand(self, rng: np.random.Generator) -> "MCTSNode":
        idx = int(rng.integers(len(self.untried)))
        col = self.untried.pop(idx)
        child = MCTSNode(self.state.transition(col), parent=self, action=col)
        self.children.append(child)
        return child

def _rollout(state: ConnectState, root_player: int,
             rng: np.random.Generator) -> float:
    cur = state
    while not cur.is_final():
        col = int(rng.choice(cur.get_free_cols()))
        cur = cur.transition(col)
    w = cur.get_winner()
    if w == 0:
        return 0.0
    return 1.0 if w == root_player else -1.0


def _mcts(root_state: ConnectState, n_sims: int, c: float,
          rng: np.random.Generator) -> int:
    root_player = root_state.player
    root = MCTSNode(root_state)

    for _ in range(n_sims):
        node = root
        while not node.untried and not node.state.is_final():
            node = node.best_child(c)

        if node.untried and not node.state.is_final():
            node = node.expand(rng)

        result = _rollout(node.state, root_player, rng)

        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value += result
            cur = cur.parent

    if not root.children:
        return int(rng.choice(root_state.get_free_cols()))

    return max(root.children, key=lambda ch: ch.visits).action

def _forced_move(state: ConnectState) -> Optional[int]:
    if state.is_final():
        return None
    for col in state.get_free_cols():
        if state.transition(col).get_winner() == state.player:
            return col
    opp = ConnectState(state.board, -state.player)
    for col in opp.get_free_cols():
        if opp.transition(col).get_winner() == opp.player:
            return col
    return None

class MCTSAgent(Policy):
    def __init__(
        self,
        n_simulations: int = 500,
        c: float = math.sqrt(2),
        use_heuristic: bool = True,
        seed: Optional[int] = None,
    ):
        self.n_simulations = n_simulations
        self.c = c
        self.use_heuristic = use_heuristic
        self.seed = seed
        self._rng: Optional[np.random.Generator] = None

    def mount(self, timeout: float = None) -> None:
        """Inicializa el RNG antes de cada partida. Acepta timeout opcional del autograder."""
        s = self.seed if self.seed is not None else int(np.random.randint(0, 2**30))
        self._rng = np.random.default_rng(s)

    def act(self, s: np.ndarray) -> int:
        if self._rng is None:
            self.mount()

        red = int(np.sum(s == -1))
        yel = int(np.sum(s == 1))
        player = -1 if red == yel else 1

        state = ConnectState(board=s, player=player)

        free = state.get_free_cols()
        if not free:
            return 0

        if self.use_heuristic:
            forced = _forced_move(state)
            if forced is not None:
                return forced

        return _mcts(state, self.n_simulations, self.c, self._rng)
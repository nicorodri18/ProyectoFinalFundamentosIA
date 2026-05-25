import numpy as np
from typing import Optional
from connect4.policy import Policy
from connect4.connect_state import ConnectState

COL_WEIGHTS = np.array([0.1, 0.2, 0.4, 0.7, 0.4, 0.2, 0.1])

def _forced_col(state: ConnectState) -> Optional[int]:
    """
    Heurística táctica 1-ply:
    1. Gana inmediatamente si puede
    2. Bloquea victoria inmediata del oponente
    """
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

def _rollout(state: ConnectState, root_player: int,
             rng: np.random.Generator, max_depth: int) -> float:
    """
    Simulación heurística hasta profundidad máxima.

    Durante el rollout las jugadas se seleccionan ponderando por
    posición estratégica (columnas centrales tienen más peso),
    sin recurrir a búsqueda 1-ply en cada paso para mantener velocidad.

    Si se alcanza el terminal: retorna +1/-1/0.
    Si se alcanza max_depth: evalúa el tablero posicionalmente.
    """
    cur = state
    for _ in range(max_depth):
        if cur.is_final():
            break
        free = cur.get_free_cols()
        weights = COL_WEIGHTS[free]
        weights = weights / weights.sum()
        col = int(rng.choice(free, p=weights))
        cur = cur.transition(col)

    if cur.is_final():
        w = cur.get_winner()
        if w == 0:
            return 0.0
        return 1.0 if w == root_player else -1.0

    board = cur.board
    score = float(np.sum(
        (board == root_player).astype(float) * COL_WEIGHTS[np.newaxis, :] -
        (board == -root_player).astype(float) * COL_WEIGHTS[np.newaxis, :]
    ))
    return float(np.tanh(score))

def _flat_monte_carlo(state: ConnectState, n_sims: int,
                      max_depth: int, rng: np.random.Generator) -> int:
    """
    Flat Monte Carlo con rollouts heurísticos posicionales.

    Evalúa cada columna válida ejecutando n_sims/n_cols simulaciones
    y selecciona la de mayor puntuación promedio.

    No usa árbol de búsqueda ni UCB — evaluación plana e independiente
    por columna candidata.
    """
    root_player = state.player
    free = state.get_free_cols()
    sims_per_col = max(1, n_sims // len(free))

    best_col = free[0]
    best_avg = -float('inf')

    for col in free:
        next_state = state.transition(col)
        if next_state.get_winner() == root_player:
            return col
        total = sum(
            _rollout(next_state, root_player, rng, max_depth)
            for _ in range(sims_per_col)
        )
        avg = total / sims_per_col
        if avg > best_avg:
            best_avg = avg
            best_col = col

    return best_col

class FlatMCTSAgent(Policy):
    """
    Agente Connect-4: Flat Monte Carlo con rollouts heurísticos posicionales.

    Diferencias clave vs MCTS clásico:
    - Sin árbol (no hay nodos, expansión ni backpropagation)
    - Sin UCB (no hay balance exploración/explotación matemático)
    - Rollouts ponderados por posición estratégica del tablero
    - Profundidad limitada con evaluación posicional al corte

    Parámetros
    ----------
    n_simulations : int
        Total de simulaciones repartidas entre columnas válidas.
    max_depth : int
        Profundidad máxima de cada rollout.
    use_heuristic : bool
        Detecta victorias/bloqueos inmediatos antes de simular.
    seed : int or None
        Semilla para reproducibilidad.
    """

    def __init__(
        self,
        n_simulations: int = 500,
        max_depth: int = 8,
        use_heuristic: bool = True,
        seed: Optional[int] = None,
    ):
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.use_heuristic = use_heuristic
        self.seed = seed
        self._rng: Optional[np.random.Generator] = None

    def mount(self, timeout: float = None) -> None:
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
            forced = _forced_col(state)
            if forced is not None:
                return forced

        return _flat_monte_carlo(state, self.n_simulations,
                                 self.max_depth, self._rng)
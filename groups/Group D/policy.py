import json
from pathlib import Path
from typing import override

import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy


class MonteCarloQAgent(Policy):
    SIMULATIONS_PER_MOVE = 5
    DRAW_REWARD = 0.5
    CENTER = ConnectState.COLS // 2
    MIN_VISITS_FOR_FULL_TRUST = 25
    DEFAULT_Q_VALUE = 0.5
    IMMEDIATE_THREAT_PENALTY = 0.35
    Q_VALUES_PATH = Path(__file__).with_name("q_values.json")

    _cached_q_values: dict[str, float] | None = None
    _cached_visit_counts: dict[str, int] | None = None

    @override
    def mount(self) -> None:
        self.rng = np.random.default_rng()
        self.q_values, self.visit_counts = self._load_q_data()

    @override
    def act(self, s: np.ndarray) -> int:
        board = np.asarray(s, dtype=int)
        state = ConnectState(board=board, player=self._infer_player(board))
        legal_actions = state.get_free_cols()

        if len(legal_actions) == 1:
            return legal_actions[0]

        winning_action = self._find_immediate_winning_move(state, state.player)
        if winning_action is not None:
            return winning_action

        blocking_action = self._find_single_blocking_move(state)
        if blocking_action is not None:
            return blocking_action

        best_action = legal_actions[0]
        best_score = -1.0

        for action in self._sorted_by_center(legal_actions):
            next_state = state.transition(action)
            score = self._score_action(state, next_state, action)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    @classmethod
    def train_q_values(
        cls,
        episodes: int = 5000,
        epsilon: float = 0.15,
        seed: int = 911,
    ) -> dict[str, int]:
        rng = np.random.default_rng(seed)
        agent = cls()
        agent.rng = rng

        q_values, visit_counts = cls._load_q_data(force_reload=True)
        wins = {-1: 0, 1: 0, 0: 0}

        for _ in range(episodes):
            state = ConnectState()
            episode_keys = {-1: [], 1: []}

            while not state.is_final():
                action = agent._select_training_action(
                    state, q_values, visit_counts, epsilon
                )
                key = agent._encode_state_action(state.board, state.player, action)
                episode_keys[state.player].append(key)
                state = state.transition(action)

            winner = state.get_winner()
            wins[winner] += 1

            rewards = {
                -1: cls.DRAW_REWARD if winner == 0 else float(winner == -1),
                1: cls.DRAW_REWARD if winner == 0 else float(winner == 1),
            }

            for player, keys in episode_keys.items():
                reward = rewards[player]
                for key in keys:
                    count = visit_counts.get(key, 0) + 1
                    old_value = q_values.get(key, cls.DEFAULT_Q_VALUE)
                    q_values[key] = old_value + (reward - old_value) / count
                    visit_counts[key] = count

        cls._save_q_data(q_values, visit_counts)
        return {
            "episodes": episodes,
            "stored_q_values": len(q_values),
            "red_wins": wins[-1],
            "yellow_wins": wins[1],
            "draws": wins[0],
        }

    def _score_action(
        self, state: ConnectState, next_state: ConnectState, action: int
    ) -> float:
        key = self._encode_state_action(state.board, state.player, action)
        learned_score = self.q_values.get(key, self.DEFAULT_Q_VALUE)
        visits = self.visit_counts.get(key, 0)
        confidence = min(visits / self.MIN_VISITS_FOR_FULL_TRUST, 1.0)

        if confidence >= 1.0:
            rollout_score = learned_score
        else:
            rollout_score = self._estimate_q_value(next_state, root_player=state.player)

        blended_score = confidence * learned_score + (1.0 - confidence) * rollout_score
        return blended_score - self._blunder_penalty(next_state)

    def _estimate_q_value(self, next_state: ConnectState, root_player: int) -> float:
        if next_state.is_final():
            winner = next_state.get_winner()
            if winner == root_player:
                return 1.0
            return self.DRAW_REWARD if winner == 0 else 0.0

        total_reward = 0.0
        for _ in range(self.SIMULATIONS_PER_MOVE):
            total_reward += self._rollout(next_state, root_player)
        return total_reward / self.SIMULATIONS_PER_MOVE

    def _rollout(self, state: ConnectState, root_player: int) -> float:
        rollout_state = ConnectState(board=state.board, player=state.player)

        while not rollout_state.is_final():
            action = self._select_rollout_action(rollout_state)
            rollout_state = rollout_state.transition(action)

        winner = rollout_state.get_winner()
        if winner == root_player:
            return 1.0
        if winner == 0:
            return self.DRAW_REWARD
        return 0.0

    def _select_rollout_action(self, state: ConnectState) -> int:
        winning_action = self._find_immediate_winning_move(state, state.player)
        if winning_action is not None:
            return winning_action

        blocking_action = self._find_single_blocking_move(state)
        if blocking_action is not None:
            return blocking_action

        legal_actions = self._safe_actions(state)
        weights = np.array(
            [ConnectState.COLS - abs(action - self.CENTER) for action in legal_actions],
            dtype=float,
        )
        probabilities = weights / weights.sum()
        return int(self.rng.choice(legal_actions, p=probabilities))

    def _select_training_action(
        self,
        state: ConnectState,
        q_values: dict[str, float],
        visit_counts: dict[str, int],
        epsilon: float,
    ) -> int:
        winning_action = self._find_immediate_winning_move(state, state.player)
        if winning_action is not None:
            return winning_action

        blocking_action = self._find_single_blocking_move(state)
        if blocking_action is not None:
            return blocking_action

        legal_actions = self._safe_actions(state)
        if self.rng.random() < epsilon:
            return int(self.rng.choice(legal_actions))

        best_action = legal_actions[0]
        best_score = -1.0

        for action in legal_actions:
            key = self._encode_state_action(state.board, state.player, action)
            score = q_values.get(key, self.DEFAULT_Q_VALUE)
            score += min(visit_counts.get(key, 0), 10) * 0.001
            score += (ConnectState.COLS - abs(action - self.CENTER)) * 0.01
            score -= self._blunder_penalty(state.transition(action))
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _find_immediate_winning_move(
        self, state: ConnectState, player: int
    ) -> int | None:
        simulated_state = ConnectState(board=state.board, player=player)
        for action in self._sorted_by_center(simulated_state.get_free_cols()):
            if simulated_state.transition(action).get_winner() == player:
                return action
        return None

    def _find_single_blocking_move(self, state: ConnectState) -> int | None:
        opponent = -state.player
        opponent_state = ConnectState(board=state.board, player=opponent)
        threats = [
            action
            for action in self._sorted_by_center(opponent_state.get_free_cols())
            if opponent_state.transition(action).get_winner() == opponent
        ]
        if len(threats) == 1:
            return threats[0]
        return None

    def _safe_actions(self, state: ConnectState) -> list[int]:
        legal_actions = self._sorted_by_center(state.get_free_cols())
        safe_actions = []
        for action in legal_actions:
            next_state = state.transition(action)
            if self._count_immediate_wins(next_state, next_state.player) == 0:
                safe_actions.append(action)
        return safe_actions if safe_actions else legal_actions

    def _blunder_penalty(self, next_state: ConnectState) -> float:
        immediate_losses = self._count_immediate_wins(next_state, next_state.player)
        return min(immediate_losses * self.IMMEDIATE_THREAT_PENALTY, 0.95)

    def _count_immediate_wins(self, state: ConnectState, player: int) -> int:
        simulated_state = ConnectState(board=state.board, player=player)
        wins = 0
        for action in simulated_state.get_free_cols():
            if simulated_state.transition(action).get_winner() == player:
                wins += 1
        return wins

    def _sorted_by_center(self, actions: list[int]) -> list[int]:
        return sorted(actions, key=lambda action: (abs(action - self.CENTER), action))

    def _encode_state_action(
        self, board: np.ndarray, player: int, action: int
    ) -> str:
        board_key = "".join(str(int(cell) + 1) for cell in board.flatten())
        return f"{player}|{action}|{board_key}"

    def _infer_player(self, board: np.ndarray) -> int:
        red_pieces = np.count_nonzero(board == -1)
        yellow_pieces = np.count_nonzero(board == 1)
        return -1 if red_pieces == yellow_pieces else 1

    @classmethod
    def _load_q_data(
        cls, force_reload: bool = False
    ) -> tuple[dict[str, float], dict[str, int]]:
        if (
            not force_reload
            and cls._cached_q_values is not None
            and cls._cached_visit_counts is not None
        ):
            return cls._cached_q_values, cls._cached_visit_counts

        if cls.Q_VALUES_PATH.exists():
            data = json.loads(cls.Q_VALUES_PATH.read_text(encoding="utf-8"))
            q_values = {key: float(value) for key, value in data["q_values"].items()}
            visit_counts = {
                key: int(value) for key, value in data["visit_counts"].items()
            }
        else:
            q_values = {}
            visit_counts = {}

        cls._cached_q_values = q_values
        cls._cached_visit_counts = visit_counts
        return q_values, visit_counts

    @classmethod
    def _save_q_data(
        cls, q_values: dict[str, float], visit_counts: dict[str, int]
    ) -> None:
        payload = {
            "q_values": q_values,
            "visit_counts": visit_counts,
        }
        cls.Q_VALUES_PATH.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        cls._cached_q_values = q_values
        cls._cached_visit_counts = visit_counts

# Agente Connect-4: MCTS + UCB1

## Idea principal

Agente basado en **Monte Carlo Tree Search (MCTS)** con selección **UCB1**, que aplica los conceptos de *online planning*, *tree search* y *multi-armed bandits* del curso.

A diferencia de un agente reactivo o basado en reglas, MCTS construye un árbol de búsqueda durante cada turno, estimando el valor de cada movida a través de simulaciones aleatorias (rollouts) hasta el estado terminal.

## Estructura del código

```
policy.py
├── MCTSNode         — nodo del árbol (estado, visitas, valor acumulado)
├── _rollout()       — simulación aleatoria hasta terminal
├── _mcts()          — loop principal: Selección → Expansión → Rollout → Backprop
├── _forced_move()   — heurística 1-ply: ganar inmediatamente o bloquear
└── MCTSAgent        — clase Policy del torneo
```

## Uso

```python
from policy import MCTSAgent

agent = MCTSAgent(
    n_simulations=500,   # más = más fuerte
    c=1.41,              # constante UCB1 (sqrt(2) por defecto)
    use_heuristic=True,  # detectar victorias/bloqueos inmediatos
    seed=42              # None para aleatorio
)
agent.mount()            # llamar antes de cada partida
col = agent.act(board)   # board es np.ndarray (6, 7)
```

## Parámetros para análisis

| Parámetro | Efecto | Valores sugeridos |
|---|---|---|
| `n_simulations` | Fuerza de juego vs velocidad | 50, 100, 200, 500, 1000 |
| `c` | Exploración UCB1 | 0.5, 1.0, √2, 2.0 |
| `use_heuristic` | Activar/desactivar 1-ply | True / False |

## Requisitos

```
numpy
```

Todas las dependencias están incluidas en el entorno del torneo.

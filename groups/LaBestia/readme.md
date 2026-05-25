# Agente Connect-4: Flat Monte Carlo con Rollouts Heurísticos

## Idea principal

Agente basado en **Flat Monte Carlo** con rollouts heurísticos posicionales.

A diferencia de MCTS clásico, este agente **no construye un árbol de búsqueda** ni usa **UCB** para balancear exploración/explotación. En cambio, evalúa cada columna válida de forma independiente ejecutando múltiples simulaciones con una política heurística que pondera posiciones estratégicas del tablero.

## Estructura del código

```
policy.py
├── COL_WEIGHTS       — pesos posicionales por columna (centro vale más)
├── _forced_col()     — heurística 1-ply: ganar inmediatamente o bloquear
├── _rollout()        — simulación heurística hasta profundidad límite
├── _flat_monte_carlo() — evalúa cada columna con n_sims simulaciones
└── FlatMCTSAgent     — clase Policy del torneo
```

## Cómo funciona

1. Si hay victoria inmediata o bloqueo necesario → lo ejecuta directamente
2. Para cada columna válida, simula `n_sims // n_columnas` partidas
3. Cada simulación avanza hasta `max_depth` movimientos, seleccionando columnas ponderadas por posición estratégica
4. Si se agota la profundidad sin llegar a terminal, evalúa el tablero posicionalmente
5. Selecciona la columna con mayor puntuación promedio acumulada

## Uso

```python
from policy import FlatMCTSAgent

agent = FlatMCTSAgent(
    n_simulations=500,  # total de simulaciones repartidas entre columnas
    max_depth=8,        # profundidad máxima de cada rollout
    use_heuristic=True, # detectar victorias/bloqueos inmediatos
    seed=42             # None para aleatorio
)
agent.mount()           # llamar antes de cada partida
col = agent.act(board)  # board es np.ndarray (6, 7)
```

## Parámetros para análisis

| Parámetro | Efecto | Valores sugeridos |
|---|---|---|
| `n_simulations` | Calidad de estimación vs velocidad | 50, 100, 200, 300, 500 |
| `max_depth` | Horizonte del rollout | 2, 4, 6, 8, 12 |
| `use_heuristic` | Activar/desactivar detección táctica 1-ply | True / False |

## Diferencias clave vs MCTS clásico

| Aspecto | Este agente | MCTS clásico |
|---|---|---|
| Árbol de búsqueda | No | Sí |
| UCB (exploración matemática) | No | Sí |
| Evaluación | Por columna independiente | Por nodo del árbol |
| Rollouts | Ponderados por posición | Aleatorios |
| Profundidad | Limitada con eval posicional | Hasta estado terminal |

## Requisitos

```
numpy
```

Todas las dependencias están incluidas en el entorno del torneo.

# MonteCarloQAgent — Connect-4

Agente inteligente para Connect-4 que combina **Q-Learning offline** con **rollouts Monte Carlo online**, desarrollado para el reto de Fundamentos de Inteligencia Artificial — Universidad de La Sabana 2026.1.

---

## Idea principal

El agente aprende valores Q para pares (estado, acción) mediante episodios de autoentrenamiento (offline). Durante el juego (online), si un estado no ha sido suficientemente visitado, complementa el valor aprendido con rollouts Monte Carlo para estimar su utilidad. Esto combina lo mejor de ambos mundos: conocimiento acumulado + exploración en tiempo real.

Lo que lo distingue de otros agentes del grupo:
- Fusión de Q-Learning tabular con estimación Monte Carlo por confianza
- Penalización de "blunders": detecta si el movimiento le da al oponente oportunidades ganadoras inmediatas
- Bloqueo de amenaza única: solo bloquea si el oponente tiene **exactamente una** amenaza (no bloquear múltiples amenazas evita distracciones)

---

## Estructura del proyecto

```
tu_carpeta/
├── monte_carlo_q_agent.py   # Código del agente
├── q_values.json            # Pesos entrenados (generado tras entrenar)
├── entrega.ipynb            # Notebook de análisis y experimentos
└── README.md                # Este archivo
```

---

## Requisitos

```bash
pip install numpy
```

El agente depende además de:
- `connect4.connect_state.ConnectState`
- `connect4.policy.Policy`

Estos módulos deben estar disponibles en el entorno del repositorio del grupo.

---

## Uso

### 1. Entrenar el agente

```python
from monte_carlo_q_agent import MonteCarloQAgent

stats = MonteCarloQAgent.train_q_values(
    episodes=5000,   # número de partidas de entrenamiento
    epsilon=0.15,    # probabilidad de exploración
    seed=911,        # semilla para reproducibilidad
)
print(stats)
# {'episodes': 5000, 'stored_q_values': ..., 'red_wins': ..., 'yellow_wins': ..., 'draws': ...}
```

Los pesos se guardan automáticamente en `q_values.json`. Las ejecuciones siguientes cargan ese archivo sin necesidad de reentrenar.

### 2. Usar el agente en una partida

```python
import numpy as np
from monte_carlo_q_agent import MonteCarloQAgent

agent = MonteCarloQAgent()
agent.mount()

board = np.zeros((6, 7), dtype=int)  # tablero vacío
action = agent.act(board)
print(f"El agente juega en la columna: {action}")
```

### 3. Parámetros configurables

| Parámetro | Default | Descripción |
|---|---|---|
| `SIMULATIONS_PER_MOVE` | 5 | Rollouts Monte Carlo por movimiento online |
| `MIN_VISITS_FOR_FULL_TRUST` | 25 | Visitas mínimas para confiar 100% en Q aprendido |
| `IMMEDIATE_THREAT_PENALTY` | 0.35 | Penalización por dejar amenazas al oponente |
| `DEFAULT_Q_VALUE` | 0.5 | Valor Q inicial para estados no visitados |
| `DRAW_REWARD` | 0.5 | Recompensa asignada a empates |

Estos se pueden modificar directamente en la clase antes de entrenar o jugar.

---

## Lógica de decisión (resumen)

```
act(board):
  1. Si solo hay un movimiento legal → jugarlo
  2. Si hay movimiento ganador inmediato → jugarlo
  3. Si el oponente tiene UNA amenaza única → bloquearla
  4. Para cada acción legal:
       score = confianza * Q_aprendido + (1-confianza) * rollout_MC - penalización_blunder
  5. Retornar la acción con mayor score
```

---

## Entrenamiento reproducible

```python
# Entrenamiento desde cero (ignora q_values.json existente)
MonteCarloQAgent.train_q_values(episodes=5000, epsilon=0.15, seed=911)

# Para experimentos con distintos recursos:
for ep in [500, 1000, 2000, 5000]:
    MonteCarloQAgent.train_q_values(episodes=ep, seed=42)
```

> **Nota:** `train_q_values` con `force_reload=True` (internamente) siempre parte desde los Q-values guardados y los incrementa. Para un entrenamiento limpio desde cero, eliminar `q_values.json` antes de llamarlo.

---

## Análisis experimental

Ver `entrega.ipynb` para:
- Win rate vs. jugador aleatorio (rojo y amarillo) en función de episodios de entrenamiento
- Desempeño del agente contra sí mismo
- Ablación: con/sin penalización de blunders, con/sin bloqueo de amenaza única
- Efecto de `SIMULATIONS_PER_MOVE` en calidad vs. tiempo de respuesta

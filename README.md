# Agente Connect-4 basado en Simulaciones Monte Carlo y UCB

Proyecto desarrollado para el curso **Fundamentos de Inteligencia Artificial**  
Universidad de La Sabana — 2026-1

---

# Descripción

Este proyecto implementa un agente inteligente para Connect-4 utilizando:

- Simulaciones Monte Carlo
- Exploration vs Exploitation
- Upper Confidence Bound (UCB)
- Heurísticas de prioridad central
- Policy Evaluation mediante rollouts

El agente utiliza múltiples simulaciones aleatorias para estimar la calidad de las acciones posibles y posteriormente selecciona movimientos utilizando una política basada en UCB.

La implementación corresponde a un enfoque simplificado basado en simulaciones Monte Carlo independientes y selección mediante estadísticas de victorias/visitas.

---

# Versiones del agente

El proyecto incluye dos variantes principales del agente:

## `backup_basic.py`

Versión inicial basada en:

- Simulaciones Monte Carlo aleatorias
- Selección mediante UCB
- Exploration vs Exploitation
- Bloqueo y victoria inmediata

Esta versión utiliza rollouts completamente aleatorios durante las simulaciones.

---

## `policy.py`

Versión mejorada del agente.

Incluye:

- Prioridad a posiciones centrales
- Rollouts guiados heurísticamente
- Bonus heurístico para columnas estratégicas
- Detección de victorias diagonales
- Simulaciones Monte Carlo mejoradas
- Selección mediante UCB

Estas mejoras permiten aumentar la calidad estratégica de las simulaciones manteniendo tiempos de ejecución bajos.

---

# Estructura del proyecto

```bash
groups/
└── Group C/
    ├── policy.py
    └── backup_basic.py
```

Archivo principal utilizado en el torneo:

```bash
groups/Group C/policy.py
```

---

# Requisitos

Instalar dependencias:

```bash
pip install numpy
```

---

# Ejecución

## Ejecutar torneo completo

Desde la raíz del proyecto:

```bash
python main.py
```

---

## Ejecutar enfrentamientos directos

```bash
python duel.py
```

---

# Configuración del agente

El número de simulaciones puede modificarse dentro del método:

```python
def act(self, s):
```

modificando la variable:

```python
simulations = 35
```

Incrementar el número de simulaciones puede mejorar la estabilidad de las decisiones, aunque también aumenta el tiempo de ejecución.

---

# Funcionamiento general

El agente sigue el siguiente flujo:

1. Obtiene movimientos válidos.
2. Detecta victorias inmediatas.
3. Bloquea victorias rivales.
4. Ejecuta simulaciones Monte Carlo.
5. Evalúa movimientos usando UCB.
6. Selecciona la mejor acción disponible.

---

# Conceptos aplicados

El proyecto aplica conceptos vistos en el curso:

- Monte Carlo Simulation
- Exploration vs Exploitation
- Upper Confidence Bound (UCB)
- Policy Evaluation
- Online Policy Improvement
- Sequential Decision Making

---

# Resultados experimentales

Durante las pruebas realizadas el agente:

- Superó consistentemente al agente aleatorio.
- Alcanzó tasas de victoria superiores al 95%.
- Mantuvo tiempos de ejecución bajos y estables.
- Mejoró su desempeño utilizando heurísticas simples.

---

# Trabajo futuro

Como posible mejora futura, podrían incorporarse:

- Simulaciones más profundas
- Heurísticas ofensivas y defensivas más avanzadas
- Evaluación posicional más sofisticada
- Técnicas híbridas con Reinforcement Learning

---

# Autor

Nicolas Rodriguez  
Fundamentos de Inteligencia Artificial  
Universidad de La Sabana

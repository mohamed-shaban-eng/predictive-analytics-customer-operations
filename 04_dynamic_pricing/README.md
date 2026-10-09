# 04 — Dynamic Pricing

This project implements both dynamic-pricing approaches mentioned in the lecture:

- Rule-Based
- Reinforcement Learning

Pricing responds to:
- demand
- inventory
- competitor price

## Rule-based model

The price is adjusted within a predefined range based on the business state.

Examples:
- high demand + low inventory → increase price
- low demand + high inventory → decrease price
- competitor much cheaper → reduce price modestly

## Reinforcement learning demo

The script includes a small Q-learning environment.

State:
- demand level
- inventory level

Actions:
- decrease price
- keep price
- increase price

Reward:
- revenue
- inventory pressure
- stockout penalty

This is an educational simulation, not a production pricing engine.

## Ethical guardrails

The lecture highlights:
- avoid price manipulation
- prevent algorithmic collusion
- maintain consumer fairness

The project therefore:
- limits the price range
- avoids customer-level discriminatory features
- does not learn from competitor coordination
- logs why a price changed

## Run

```bash
python dynamic_pricing.py
```

Outputs:
```text
outputs/dynamic_pricing_simulation.csv
outputs/q_learning_policy.csv
```

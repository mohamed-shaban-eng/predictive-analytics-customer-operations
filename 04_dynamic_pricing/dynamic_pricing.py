from pathlib import Path

import numpy as np
import pandas as pd

# in this code i used reinforcdment learning algorithme but i build it by my self not import it as pretrained

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def rule_based_price(
    base_price: float,
    demand: float,
    inventory: int,
    competitor_price: float,
    min_price_multiplier: float = 0.80,
    max_price_multiplier: float = 1.25,
):
    """
    Rule-based dynamic pricing using exactly the business signals
    emphasized in the lecture:
    demand, inventory, competitors.

    Returns:
        new_price, reason
    """
    multiplier = 1.0
    reasons = []

    # Demand signal
    if demand >= 0.80:
        multiplier += 0.10
        reasons.append("high demand")
    elif demand <= 0.30:
        multiplier -= 0.08
        reasons.append("low demand")

    # Inventory signal
    if inventory <= 20:
        multiplier += 0.08
        reasons.append("low inventory")
    elif inventory >= 80:
        multiplier -= 0.07
        reasons.append("high inventory")

    # Competitor signal
    competitor_ratio = competitor_price / base_price

    if competitor_ratio <= 0.90:
        multiplier -= 0.05
        reasons.append("competitor cheaper")
    elif competitor_ratio >= 1.10:
        multiplier += 0.04
        reasons.append("competitor more expensive")

    # Ethical/bounded price movement
    multiplier = np.clip(
        multiplier,
        min_price_multiplier,
        max_price_multiplier,
    )

    new_price = round(base_price * multiplier, 2)

    if not reasons:
        reasons.append("stable market")

    reason = "; ".join(reasons)
    return new_price, reason


class SimplePricingEnvironment:
    """
    Small educational environment for Q-learning.

    State:
        demand level: low / medium / high
        inventory level: low / medium / high

    Actions:
        -1 = decrease price
         0 = keep price
        +1 = increase price
    """

    DEMAND_LEVELS = [0, 1, 2]
    INVENTORY_LEVELS = [0, 1, 2]
    ACTIONS = [-1, 0, 1]

    def __init__(self, seed: int = RANDOM_STATE):
        self.rng = np.random.default_rng(seed)
        self.base_price = 100.0
        self.reset()

    def reset(self):
        self.demand_level = int(self.rng.integers(0, 3))
        self.inventory_level = int(self.rng.integers(0, 3))
        return (self.demand_level, self.inventory_level)

    def step(self, action: int):
        """
        Return:
            next_state, reward
        """
        # Convert levels into demand probabilities
        demand_strength = {
            0: 0.20,
            1: 0.55,
            2: 0.85,
        }[self.demand_level]

        # Convert actions into price multiplier
        price_multiplier = {
            -1: 0.90,
            0: 1.00,
            1: 1.10,
        }[action]

        price = self.base_price * price_multiplier

        # Demand response to price
        adjusted_probability = np.clip(
            demand_strength * (1.05 if action == -1 else 1.0),
            0.05,
            0.95,
        )

        units_sold = int(
            self.rng.binomial(
                n=10,
                p=adjusted_probability,
            )
        )

        # Inventory level changes after sales
        inventory_units = {
            0: 20,
            1: 60,
            2: 100,
        }[self.inventory_level]

        units_sold = min(units_sold, inventory_units)

        revenue = units_sold * price

        stockout_penalty = 0.0
        if inventory_units <= units_sold and demand_strength > 0.70:
            stockout_penalty = 100.0

        reward = revenue - stockout_penalty

        # Next inventory level
        remaining = inventory_units - units_sold

        if remaining < 30:
            next_inventory = 0
        elif remaining < 70:
            next_inventory = 1
        else:
            next_inventory = 2

        # Next demand is stochastic
        next_demand = int(
            self.rng.integers(0, 3)
        )

        next_state = (next_demand, next_inventory)

        self.demand_level = next_demand
        self.inventory_level = next_inventory

        return next_state, reward


def train_q_learning(
    episodes: int = 3000,
    alpha: float = 0.10,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.05,
):
    """
    Train a small tabular Q-learning pricing policy.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    states = [
        (demand, inventory)
        for demand in range(3)
        for inventory in range(3)
    ]

    actions = SimplePricingEnvironment.ACTIONS

    q_table = {
        state: np.zeros(len(actions), dtype=float)
        for state in states
    }

    env = SimplePricingEnvironment()

    for _ in range(episodes):
        state = env.reset()

        for _ in range(30):
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action_index = int(
                    rng.integers(0, len(actions))
                )
            else:
                action_index = int(
                    np.argmax(q_table[state])
                )

            action = actions[action_index]

            next_state, reward = env.step(action)

            best_next = np.max(q_table[next_state])

            q_table[state][action_index] = (
                q_table[state][action_index]
                + alpha
                * (
                    reward
                    + gamma * best_next
                    - q_table[state][action_index]
                )
            )

            state = next_state

        epsilon = max(
            min_epsilon,
            epsilon * epsilon_decay,
        )

    return q_table


def main():
    print("=" * 70)
    print("DYNAMIC PRICING - RULE BASED + Q-LEARNING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Rule-based simulation
    # --------------------------------------------------------
    rng = np.random.default_rng(RANDOM_STATE)

    rows = []

    base_price = 100.0

    for day in range(1, 31):
        demand = float(rng.uniform(0.10, 0.95))
        inventory = int(rng.integers(10, 110))
        competitor_price = float(
            base_price * rng.uniform(0.88, 1.15)
        )

        price, reason = rule_based_price(
            base_price=base_price,
            demand=demand,
            inventory=inventory,
            competitor_price=competitor_price,
        )

        rows.append(
            {
                "day": day,
                "demand_index": round(demand, 3),
                "inventory": inventory,
                "competitor_price": round(competitor_price, 2),
                "recommended_price": price,
                "reason": reason,
            }
        )

    simulation_df = pd.DataFrame(rows)

    simulation_path = OUTPUT_DIR / "dynamic_pricing_simulation.csv"
    simulation_df.to_csv(
        simulation_path,
        index=False,
    )

    print("\nRule-based pricing examples:")
    print(simulation_df.head(10).to_string(index=False))

    # --------------------------------------------------------
    # 2. Q-learning
    # --------------------------------------------------------
    print("\nTraining Q-learning pricing policy...")

    q_table = train_q_learning()

    policy_rows = []

    action_names = {
        -1: "decrease_price",
        0: "keep_price",
        1: "increase_price",
    }

    actions = SimplePricingEnvironment.ACTIONS

    for state, values in q_table.items():
        best_index = int(np.argmax(values))
        best_action = actions[best_index]

        policy_rows.append(
            {
                "demand_level": state[0],
                "inventory_level": state[1],
                "best_action": action_names[best_action],
                "q_value": round(float(values[best_index]), 4),
            }
        )

    policy_df = pd.DataFrame(policy_rows)

    policy_path = OUTPUT_DIR / "q_learning_policy.csv"
    policy_df.to_csv(
        policy_path,
        index=False,
    )

    print("\nLearned Q-learning policy:")
    print(policy_df.to_string(index=False))

    print("\nSaved:")
    print(simulation_path)
    print(policy_path)


if __name__ == "__main__":
    main()

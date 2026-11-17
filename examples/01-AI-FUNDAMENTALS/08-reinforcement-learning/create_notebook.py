"""Script to generate and execute the Reinforcement Learning companion notebook.

This script uses nbformat to create reinforcement_learning.ipynb and
nbclient to execute all cells, populating actual outputs.
"""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

DIR = Path(__file__).parent
OUTPUT_NOTEBOOK = DIR / "reinforcement_learning.ipynb"

def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.5"
        }
    }

    cells = []

    # Cell 1: Markdown Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# Reinforcement Learning from Scratch: Tabular Q-Learning in GridWorld

Welcome to the hands-on companion notebook for **Reinforcement Learning** in the AI Agent Engineering Academy.

In this notebook, you will build a complete reinforcement learning system using **only the Python standard library** — no heavyweight game engines (`gym`/`gymnasium`) or complex C-extensions required.

---

### What You Will Build:
1. **4x4 GridWorld Environment**: A discrete grid environment with a start position, obstacles/traps, step penalties, and a terminal goal.
2. **Tabular Q-Learning Agent**: An agent that maintains a Q-table mapping state-action pairs $(s, a)$ to expected cumulative discounted rewards.
3. **Epsilon-Greedy Exploration**: A balancing mechanism that explores random actions early on and exploits high-value learned paths later.
4. **Training Loop & Reward Curve**: Training the agent across 600 episodes and observing convergence from random stumbling to optimal navigation.
5. **Policy Extraction & ASCII Visualization**: Visualizing the Q-values, the optimal policy arrows, and verifying the exact shortest-path trajectory.

Let's inspect our Python environment first.
"""))

    # Cell 2: Code - Python Environment Check
    cells.append(nbf.v4.new_code_cell("""import sys
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

print(f"Python version: {sys.version}")
print("Pure Python standard library modules loaded successfully.")
"""))

    # Cell 3: Markdown - Step 1: Environment Design
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 1: Building the Custom 4x4 GridWorld Environment

A reinforcement learning task requires an **Environment** that the agent interacts with according to the Markov Decision Process (MDP) formulation:
- **State Space ($S$):** A 4x4 grid consisting of 16 distinct states $(0 \dots 15)$, corresponding to coordinates $(r, c)$ where $r, c \in \{0, 1, 2, 3\}$.
- **Action Space ($A$):** 4 discrete actions: `0: UP (-1, 0)`, `1: DOWN (+1, 0)`, `2: LEFT (0, -1)`, `3: RIGHT (0, +1)`.
- **Start State:** $(0, 0)$ (top-left corner).
- **Goal State:** $(3, 3)$ (bottom-right corner) with reward $+10.0$ (terminal).
- **Traps / Obstacles:** $(1, 1)$ and $(2, 2)$ with penalty $-10.0$ (terminal).
- **Step Cost:** $-0.1$ for each valid move to encourage the agent to find the shortest possible path rather than wandering aimlessly.
- **Boundaries:** Attempting to walk off the edge bounces back into the same cell with a step cost of $-0.1$.
"""))

    # Cell 4: Code - GridWorld Environment Implementation
    cells.append(nbf.v4.new_code_cell("""class GridWorld:
    ACTIONS = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1)    # RIGHT
    }
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
    ACTION_ARROWS = ["^", "v", "<", ">"]

    def __init__(self, size: int = 4, start: Tuple[int, int] = (0, 0),
                 goal: Tuple[int, int] = (3, 3), traps: List[Tuple[int, int]] = None):
        self.size = size
        self.start = start
        self.goal = goal
        self.traps = traps if traps is not None else [(1, 1), (2, 2)]
        self.state = start

    def reset(self) -> int:
        \"\"\"Reset agent to the start state and return state index.\"\"\"
        self.state = self.start
        return self.to_index(self.state)

    def to_index(self, pos: Tuple[int, int]) -> int:
        \"\"\"Convert (row, col) coordinates to integer state index (0 to size*size - 1).\"\"\"
        return pos[0] * self.size + pos[1]

    def to_pos(self, index: int) -> Tuple[int, int]:
        \"\"\"Convert integer state index to (row, col) coordinates.\"\"\"
        return (index // self.size, index % self.size)

    def step(self, action: int) -> Tuple[int, float, bool]:
        \"\"\"Execute an action and return (next_state_index, reward, done).\"\"\"
        dr, dc = self.ACTIONS[action]
        # Boundary constraints
        new_row = max(0, min(self.size - 1, self.state[0] + dr))
        new_col = max(0, min(self.size - 1, self.state[1] + dc))
        self.state = (new_row, new_col)
        next_index = self.to_index(self.state)

        # Terminal conditions & rewards
        if self.state == self.goal:
            return next_index, 10.0, True
        elif self.state in self.traps:
            return next_index, -10.0, True
        else:
            return next_index, -0.1, False

    def render(self) -> str:
        \"\"\"Render the grid world as an ASCII board.\"\"\"
        lines = []
        for r in range(self.size):
            row_str = []
            for c in range(self.size):
                pos = (r, c)
                if pos == self.state and pos == self.start:
                    row_str.append("[S]")
                elif pos == self.state:
                    row_str.append("[A]")
                elif pos == self.start:
                    row_str.append(" S ")
                elif pos == self.goal:
                    row_str.append(" G ")
                elif pos in self.traps:
                    row_str.append(" X ")
                else:
                    row_str.append(" . ")
            lines.append(" ".join(row_str))
        return "\\n".join(lines)

# Quick verification of GridWorld mechanics
env = GridWorld()
start_idx = env.reset()
print("Initial Grid Layout (S=Start, G=Goal, X=Trap, .=Empty):")
print(env.render())
print(f"\\nStart state index: {start_idx} (coordinates: {env.to_pos(start_idx)})")

# Test action: move RIGHT (action 3)
next_s, reward, done = env.step(3)
print(f"After moving RIGHT -> next_state: {next_s} ({env.to_pos(next_s)}), reward: {reward}, done: {done}")
assert next_s == 1
assert reward == -0.1
assert not done
print("Quick check PASSED: GridWorld transition and step mechanics work correctly.")
"""))

    # Cell 5: Markdown - Step 2: Q-Learning Agent
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 2: Implementing the Q-Learning Agent

**Q-Learning** is a model-free, off-policy Temporal Difference (TD) reinforcement learning algorithm developed by Chris Watkins (1989).

### The Q-Learning Update Rule:
$$Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]$$

Where:
- $Q(s, a)$ is the expected total future discounted reward for taking action $a$ in state $s$.
- $\\alpha$ is the **learning rate** ($0 < \\alpha \\le 1$), controlling how much new experience overwrites old estimates.
- $r$ is the immediate **reward** received after taking action $a$.
- $\\gamma$ is the **discount factor** ($0 \\le \\gamma < 1$), weighing immediate rewards against future rewards.
- $\\max_{a'} Q(s', a')$ is the agent's estimate of the optimal future value achievable from the next state $s'$.
- $\\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]$ is the **TD Error** (the discrepancy between the target estimate and the current estimate).

### Exploration vs. Exploitation: Epsilon-Greedy
- With probability $\\epsilon$, pick a uniformly random action (**Exploration**).
- With probability $1 - \\epsilon$, pick $\\arg\\max_{a} Q(s, a)$ (**Exploitation**).
- $\\epsilon$ decays over episodes from $\\epsilon_{\\text{start}} = 1.0$ down to $\\epsilon_{\\text{min}} = 0.05$.
"""))

    # Cell 6: Code - QLearningAgent Implementation
    cells.append(nbf.v4.new_code_cell("""class QLearningAgent:
    def __init__(self, num_states: int, num_actions: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table with zeros: size [num_states][num_actions]
        self.q_table = [[0.0 for _ in range(num_actions)] for _ in range(num_states)]

    def choose_action(self, state: int) -> int:
        \"\"\"Select action using epsilon-greedy strategy.\"\"\"
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        # Greedy choice: pick action with highest Q-value in this state
        return max(range(self.num_actions), key=lambda a: self.q_table[state][a])

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        \"\"\"Perform Bellman TD update on Q-table.\"\"\"
        current_q = self.q_table[state][action]
        if done:
            target = reward  # Terminal state has zero future value
        else:
            best_future_q = max(self.q_table[next_state])
            target = reward + self.gamma * best_future_q

        # TD update
        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        \"\"\"Decay epsilon exploration rate towards minimum bound.\"\"\"
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

print("QLearningAgent class defined with tabular storage and epsilon-greedy policy.")
"""))

    # Cell 7: Markdown - Step 3: Training Loop
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 3: Training the Q-Learning Agent

We train the agent over **600 episodes**. In each episode:
1. Reset the environment to start position $(0, 0)$.
2. Take steps until the episode ends (reaches Goal, falls into a Trap, or hits max 50 steps limit).
3. At each step, update the Q-table via TD learning.
4. Decay epsilon after each episode.
5. Record total rewards and steps taken.
"""))

    # Cell 8: Code - Running the Training Loop
    cells.append(nbf.v4.new_code_cell("""# Set fixed random seed for reproducibility
random.seed(42)

env = GridWorld()
num_states = env.size * env.size
num_actions = len(env.ACTIONS)

agent = QLearningAgent(
    num_states=num_states,
    num_actions=num_actions,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.99
)

NUM_EPISODES = 600
MAX_STEPS = 50

rewards_history = []
steps_history = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0.0
    step_count = 0
    done = False

    while not done and step_count < MAX_STEPS:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        step_count += 1

    agent.decay_epsilon()
    rewards_history.append(total_reward)
    steps_history.append(step_count)

print(f"Training completed over {NUM_EPISODES} episodes.")
assert len(rewards_history) == NUM_EPISODES
assert len(steps_history) == NUM_EPISODES
"""))

    # Cell 9: Markdown - Step 4: Evaluating Learning Progress
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 4: Evaluating Learning Progress and Reward Trends

To quantify learning, we evaluate performance across consecutive 100-episode blocks.
Notice the sharp improvement:
- **Episodes 1-100**: Low or negative average rewards as the agent stumbles randomly and triggers traps.
- **Episodes 501-600**: High positive average reward near $+9.5$, with minimal steps taken to reach the goal.
"""))

    # Cell 10: Code - Summary Statistics & ASCII Reward Chart
    cells.append(nbf.v4.new_code_cell("""print("=" * 65)
print(f"{'Episode Block':<18} | {'Avg Reward':<12} | {'Avg Steps':<10} | {'Epsilon':<8}")
print("=" * 65)

block_size = 100
for i in range(0, NUM_EPISODES, block_size):
    block_rewards = rewards_history[i:i + block_size]
    block_steps = steps_history[i:i + block_size]
    avg_r = sum(block_rewards) / len(block_rewards)
    avg_s = sum(block_steps) / len(block_steps)
    # Epsilon at the end of this block
    eps_at_block = max(0.05, 1.0 * (0.99 ** (i + block_size)))
    print(f"Episodes {i+1:3d}-{i+block_size:<4d}  | {avg_r:>10.2f}   | {avg_s:>8.2f}   | {eps_at_block:>6.4f}")

first_50_avg = sum(rewards_history[:50]) / 50
last_50_avg = sum(rewards_history[-50:]) / 50

print("=" * 65)
print(f"First 50 episodes average reward: {first_50_avg:.2f}")
print(f"Last 50 episodes average reward:  {last_50_avg:.2f}")

# Quick-check assertion
assert last_50_avg > first_50_avg, "Agent failed to improve performance over time!"
assert last_50_avg > 8.0, f"Expected final average reward > 8.0, got {last_50_avg:.2f}"
print("\\nQuick check PASSED: Agent successfully converged to high positive reward.")
"""))

    # Cell 11: Code - ASCII Reward Curve
    cells.append(nbf.v4.new_code_cell("""def render_ascii_curve(history: List[float], bins: int = 12):
    \"\"\"Render an ASCII bar chart of average rewards over training bins.\"\"\"
    bin_size = len(history) // bins
    print("\\nReward Improvement Trend (ASCII Visualization):")
    print("-" * 55)
    for b in range(bins):
        chunk = history[b * bin_size : (b + 1) * bin_size]
        avg = sum(chunk) / len(chunk)
        # Normalize -12.0 to +10.0 range to bar width 0-30 chars
        bar_len = int(max(0, min(30, (avg + 12.0) / 22.0 * 30)))
        bar = "#" * bar_len
        ep_label = f"Ep {b*bin_size+1:3d}-{(b+1)*bin_size:3d}"
        print(f"{ep_label} | {avg:>6.2f} | {bar}")
    print("-" * 55)

render_ascii_curve(rewards_history, bins=12)
"""))

    # Cell 12: Markdown - Step 5: Visualizing Q-Table and Optimal Policy
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 5: Inspecting the Learned Q-Table and Optimal Policy

After training, the **optimal policy** $\\pi^*(s)$ is derived greedily from the Q-table:
$$\\pi^*(s) = \\arg\\max_{a} Q(s, a)$$

Let's display:
1. The **Best Action Arrows** for each cell in the 4x4 grid:
   - `^` : UP
   - `v` : DOWN
   - `<` : LEFT
   - `>` : RIGHT
   - `X` : Trap
   - `G` : Goal
2. The **Maximum Q-Value** for each state.
"""))

    # Cell 13: Code - Display Policy Grid and Max Q-values
    cells.append(nbf.v4.new_code_cell("""print("Learned Policy Grid (Directions to follow):")
print("-" * 35)

policy_grid = []
for r in range(env.size):
    row_symbols = []
    for c in range(env.size):
        pos = (r, c)
        s_idx = env.to_index(pos)
        if pos == env.goal:
            row_symbols.append(" G ")
        elif pos in env.traps:
            row_symbols.append(" X ")
        else:
            best_a = max(range(num_actions), key=lambda a: agent.q_table[s_idx][a])
            arrow = env.ACTION_ARROWS[best_a]
            row_symbols.append(f" {arrow} ")
    print(" ".join(row_symbols))
    policy_grid.append(row_symbols)

print("-" * 35)

print("\\nMax Q-Values Grid (Expected cumulative returns):")
print("-" * 35)
for r in range(env.size):
    row_q = []
    for c in range(env.size):
        pos = (r, c)
        s_idx = env.to_index(pos)
        if pos == env.goal:
            row_q.append(f"  GOAL ")
        elif pos in env.traps:
            row_q.append(f"  TRAP ")
        else:
            max_q = max(agent.q_table[s_idx])
            row_q.append(f"{max_q:>6.2f} ")
    print(" ".join(row_q))
print("-" * 35)
"""))

    # Cell 14: Markdown - Step 6: Testing the Greedy Trajectory
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 6: Evaluating the Optimal Greedy Trajectory

Now we test the trained agent in pure **exploitation mode** (setting exploration $\\epsilon = 0.0$).
We record the exact sequence of states visited from $(0, 0)$ to the Goal $(3, 3)$.
"""))

    # Cell 15: Code - Greedy Evaluation Run
    cells.append(nbf.v4.new_code_cell("""# Test greedy navigation without exploration
agent.epsilon = 0.0

eval_env = GridWorld()
current_s = eval_env.reset()
trajectory = [eval_env.state]
total_eval_reward = 0.0
eval_steps = 0
done = False

print("Executing trained agent greedy path:")
while not done and eval_steps < 20:
    action = agent.choose_action(current_s)
    act_name = eval_env.ACTION_NAMES[action]
    next_s, reward, done = eval_env.step(action)
    total_eval_reward += reward
    eval_steps += 1
    trajectory.append(eval_env.state)
    print(f"Step {eval_steps}: Took {act_name:<5} -> reached {eval_env.state}, reward: {reward:+.1f}")
    current_s = next_s

print("\\nEvaluation Summary:")
print(f"Start Position:       {trajectory[0]}")
print(f"Goal Reached:         {trajectory[-1] == eval_env.goal}")
print(f"Full Trajectory:      {trajectory}")
print(f"Total Steps:          {eval_steps}")
print(f"Total Return:         {total_eval_reward:.2f}")

# Quick-check assertions
assert trajectory[0] == (0, 0), "Agent did not start at (0, 0)"
assert trajectory[-1] == (3, 3), "Agent failed to reach goal at (3, 3)"
assert (1, 1) not in trajectory, "Agent stepped into Trap at (1, 1)!"
assert (2, 2) not in trajectory, "Agent stepped into Trap at (2, 2)!"
assert eval_steps == 6, f"Optimal path should take exactly 6 steps, took {eval_steps}"
assert round(total_eval_reward, 2) == 9.5, f"Expected 9.5 reward, got {total_eval_reward}"
print("\\nAll quick-check assertions PASSED successfully!")
"""))

    # Cell 16: Markdown - Step 7: Connection to AI Agents
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 7: How Reinforcement Learning Connects to AI Agents

Reinforcement Learning is the foundational engine behind modern advanced AI Agent Engineering:

1. **Reinforcement Learning from Human Feedback (RLHF)**:
   Modern LLMs (such as GPT-4, Claude, and Llama) use Proximal Policy Optimization (PPO) or Direct Preference Optimization (DPO) to align raw next-token prediction with human preferences, safety standards, and helpfulness.
2. **Tool-Use and Reasoning Agents (ReAct / Tree-of-Thoughts)**:
   When an autonomous agent decides whether to search the web, query a database, or execute Python code, each tool invocation acts as an **Action** $a_t$, the tool output and conversation context form the **State** $s_t$, and task success acts as the **Reward** $r_t$.
3. **Credit Assignment over Long Sequences**:
   Just as our GridWorld agent received a reward only at the end and learned to propagate values backward using Temporal Difference error, autonomous agents must solve credit assignment across multi-step research and workflow pipelines.
"""))

    nb.cells = cells
    return nb

def main():
    print("Building notebook structure...")
    nb = build_notebook()
    
    print(f"Writing initial notebook to {OUTPUT_NOTEBOOK}...")
    with open(OUTPUT_NOTEBOOK, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("Executing notebook with nbclient to populate actual outputs...")
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(DIR)}}
    )
    client.execute()

    print(f"Saving notebook with outputs to {OUTPUT_NOTEBOOK}...")
    with open(OUTPUT_NOTEBOOK, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("Done! Notebook built and executed successfully.")

if __name__ == "__main__":
    main()

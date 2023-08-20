# Pac-Man AI Project (2021)

This repository showcases the Third phase of Pac-Man AI Project developed as part of the "Principles and Applications of Artificial Intelligence" course in 2021. The project aims to utilize basic AI methods to develop solutions for the classic Pac-Man game. You can find the links to phase two and three below.

## Project Overview

- **Goal:** Utilize AI methods to enhance Pac-Man's performance and decision-making in the game.
- **Implementation:** The project is implemented in Python.
- **Course:** Principles and Applications of Artificial Intelligence.
- **Grade:** 20/20
- **Professor:** Dr. Javanmardi

## Implemented Algorithms

The project explores various AI algorithms to tackle different aspects of Pac-Man's gameplay:

### First Phase: [GitHub Repository](https://github.com/leilibrk/Pacman-AI-Project)

- **Uninformed Search Algorithms:** Implemented DFS, BFS, and UCS algorithms for efficient traversal and consumption of all nodes while keeping Pac-Man alive.
- **Informed Search Algorithms:** Utilized the A* algorithm for optimized pathfinding and decision-making.

### Second Phase: [GitHub Repository](https://github.com/leilibrk/Pacman-multiAgent)

- **Adversarial Search Algorithms:** Designed strategies for Pac-Man to reach the best utility while avoiding ghosts in a multiagent scenario.
- **Minimax with Alpha-Beta Pruning:** Implemented the minimax algorithm with alpha-beta pruning for better performance in adversarial scenarios.
- **Expectimax:** Developed the Expectimax algorithm to handle uncertainty in decision-making.

### Third Phase: 
- **Model-Free Reinforcement Learning:** Implemented the Q-learning algorithm, a model-free reinforcement learning approach, to optimize Pac-Man's actions.

## Pacman Multi-Agent results

| Method                             | Average Score | Win Rate | Runtime     |
|------------------------------------|---------------|----------|-------------|
| Minimax                            | -92           | 40%      | 42.23 sec   |
| Minimax with alpha-beta pruning    | -80           | 40%      | 33.5 sec    |
| Expectimax                         | 206.8         | 70%      | 46.27 sec   |

## Reinforcement Learning Model Results: 
<img src="https://github.com/leilibrk/Pacman-AI-Project/blob/master/pacman-RL.png" title="" alt="zip2.png" width="300">

## Acknowledgments

We extend our gratitude to Dr. Javanmardi for guiding us through the course and inspiring us to tackle the challenges of Pac-Man using AI methods.

## Contributing

Contributions to this repository are welcome! Feel free to fork the repository and submit pull requests for improvements, optimizations, or extensions.

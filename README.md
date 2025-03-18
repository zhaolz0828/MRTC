---

# 🚀 Dynamic Task Assignment and Single Unmanned Sanitation Vehicle (USV) Path Planning 🔍

This repository contains Python implementations of two advanced algorithms:  
1. **Dynamic Task Assignment Algorithm** based on Actor-Critic Reinforcement Learning (RL).  
2. **Single Unmanned Sanitation Vehicle (USV) Path Planning Algorithm** inspired by Ant Colony Optimization (ACO).  

These algorithms are designed to solve complex optimization problems in the context of connected road segments and unmanned sanitation vehicle (USV) path planning.

---

## 📋 Table of Contents

1. [Dynamic Task Assignment Algorithm](#1-dynamic-task-assignment-algorithm)
2. [Single USV Path Planning Algorithm](#2-single-unmanned-sanitation-vehicle-usv-path-planning-algorithm)

---

## 1. Dynamic Task Assignment Algorithm

### 🎯 Overview

The dynamic task assignment algorithm uses an **Actor-Critic Reinforcement Learning (RL)** framework to assign tasks to
connected road segments. The goal is to optimize task assignment by minimizing distances, incorporating factors such as
shortest paths, water refill station distances, and warehouse distances.

### ✨ Key Features

- **Euclidean Distance Calculation**: Computes the distance between two coordinates.
- **Greedy Lowest-Cost Path Algorithm**: Approximates the lowest-cost path in a graph using a greedy approach.
- **Actor-Critic RL Framework**:
    - **Actor Network**: Generates a probability distribution over actions (task assignments).
    - **Critic Network**: Estimates the value of a given state to guide the actor.
- **Dynamic Task Assignment**: Iteratively updates task assignments based on rewards and penalties.

### 🛠️ Implementation

The main function `dynamic_task_assignment` performs the task assignment. It takes the following inputs:

| Parameter          | Description                                         |
|--------------------|-----------------------------------------------------|
| `M`                | Adjacency matrix of connected road segments.        |
| `num_tasks`        | Number of tasks to assign.                          |
| `num_episodes`     | Number of training episodes.                        |
| `node_indices`     | Indices of connected road segments.                 |
| `road_coordinates` | Coordinates of connected road segments.             |
| `site_coordinates` | Coordinates of sites (e.g., water refill stations). |

---

## 2. Single Unmanned Sanitation Vehicle (USV) Path Planning Algorithm

### 🎯 Overview

The Single Unmanned Sanitation Vehicle (USV) Path Planning Algorithm uses a virtual agent-based approach inspired by Ant
Colony Optimization (ACO). It aims to find the lowest-cost path for a USV while considering constraints such as water
consumption, travel range, and task execution time limits.

### ✨ Key Features

- **Pheromone-Based Path Exploration**: Virtual agents explore paths and deposit pheromones to guide subsequent agents.
- **Constraints Handling**:
    - Water capacity and travel range limits.
    - Task execution time limits.
- **Optimization**: Minimizes the total cost while ensuring compliance with all constraints..

### 🛠️ Implementation

The algorithm is implemented using a virtual agent-based approach. Below is a high-level overview of the steps:

1. **Initialize Pheromone Levels**: Set initial pheromone values for all paths.
2. **Virtual Agent Exploration**: Agents explore paths based on pheromone levels and constraints.
3. **Pheromone Update**: Update pheromone levels based on the quality of paths explored.
4. **Path Selection**: Select the lowest-cost path that satisfies all constraints.





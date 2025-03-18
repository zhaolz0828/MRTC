"""This script implements a dynamic task assignment algorithm based on the Actor - Critic Reinforcement Learning (RL) method,
# aiming to solve the task assignment problem for connected road segments.
# 1. Defines a function to calculate the Euclidean distance between two points.
# 2. Implements a greedy algorithm to find an approximate shortest path in a graph.
# 3. Constructs Actor and Critic neural networks for RL.
# 4. Implements training and updating of the Actor - Critic RL model.
"""

def calculate_distance(coord1, sitecoord, site_coordinates):
    """
    Calculate Euclidean distance between two coordinates.
    Args:
        coord1 (tuple): First coordinate (x1, y1).
        sitecoord (tuple): Second coordinate (x2, y2).
    Returns:
        float: Euclidean distance between coord1 and coord2.
    """
    x1, y1 = coord1
    x2, y2 = site_coordinates[sitecoord]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def greedy_shortest_path(sub_task_matrix):
    """
    This function implements a greedy algorithm to find an approximate shortest path
    in a complete graph represented by a distance matrix. The greedy algorithm starts
    from a given connected road segment (in this case, connected road segments 0) and at each step, it selects the nearest
    unvisited connected road segment to the current connected road segment.

    Parameters:
    sub_task_matrix (numpy.ndarray): A square matrix representing the distances between connected road segments.
                               Each element sub_task_matrix[i][j] represents the distance from connected road segment i to connected road segment j.

    Returns:
    float: The total length of the path found by the greedy algorithm.
    """
    num_nodes = sub_task_matrix.shape[0]
    visited = [False] * num_nodes
    path_length = 0
    current_node = 0
    for _ in range(num_nodes - 1):
        visited[current_node] = True
        distances = sub_task_matrix[current_node]
        min_distance = np.inf
        next_node = -1
        for node in range(num_nodes):
            if not visited[node] and distances[node] < min_distance:
                min_distance = distances[node]
                next_node = node
        path_length += min_distance
        current_node = next_node
    return path_length


class ActorNetwork(nn.Module):
    """
    The ActorNetwork class is designed as a neural network module in PyTorch, which serves as the actor
    in an Actor - Critic reinforcement learning framework. Its main function is to generate a probability
    distribution over actions (clusters in this context) given an input state.

    Parameters:
    - num_states (int): The number of features in the input state, which determines the input dimension of the network.
    - num_clusters (int): The number of possible actions or clusters, which determines the output dimension of the network.
    """

    def __init__(self, num_states, num_clusters):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(num_states, 256)
        self.fc2 = nn.Linear(256, num_clusters)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x


class CriticNetwork(nn.Module):
    """
    The CriticNetwork class is a neural network module in PyTorch, acting as the critic
    in an Actor - Critic reinforcement learning framework. It estimates the value of a given state.

    Parameters:
    - num_states (int): The number of features in the input state, which determines the input dimension of the network.
    """

    def __init__(self, num_states):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(num_states, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ActorCriticRL:
    def __init__(self, num_tasks, num_states):
        self.num_tasks = num_tasks
        self.num_states = num_states

        # Initialize actor and critic networks
        self.actor = ActorNetwork(num_states, num_tasks)
        self.critic = CriticNetwork(num_states)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0003)

        # Hyperparameters
        self.discount_factor = 0.2
        self.exploration_rate = 0.5
        self.gradient_clip_value = 1

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor).detach().numpy().flatten()

        # Check for NaN values
        if np.isnan(action_probs).any():
            print("NaN detected in action probabilities")
            action_probs = np.nan_to_num(action_probs, nan=1.0 / self.num_clusters)

        return np.random.choice(self.num_clusters, p=action_probs)

    def update_networks(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])

        # Compute value targets
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)
        td_error = reward_tensor + self.discount_factor * next_value - value

        # Update critic
        critic_loss = td_error.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip_value)  # Gradient clipping
        self.critic_optimizer.step()

        # Update actor
        action_tensor = torch.LongTensor([action])
        log_prob = torch.log(self.actor(state_tensor).gather(1, action_tensor.unsqueeze(1)))
        actor_loss = -(log_prob * td_error.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_value)  # Gradient clipping
        self.actor_optimizer.step()


def dynamic_task_assignment(M, num_tasks, num_episodes, node_indices, road_coordinates, site_coordinates):
    """
    Perform dynamic task assignment using an Actor - Critic RL algorithm.

    Parameters:
    M (np.ndarray): Adjacency matrix of connected road segments.
    num_tasks (int): Number of tasks.
    num_episodes (int): Number of training episodes.
    node_indices (np.ndarray): Indices of connected road segments.
    road_coordinates (dict): Coordinates of connected road segments.
    site_coordinates (list): Coordinates of sites.

    Returns:
    np.ndarray: Task labels for each connected road segment.
    """
    num_states = len(M)

    # Normalize the adjacency matrix
    scaler = StandardScaler()
    M = scaler.fit_transform(M)

    warehouse_distance = 0
    warehouse_coordinate = [-1500.0, -1500.0]

    ac_rl_model = ActorCriticRL(num_tasks, num_states)

    # Initialize centroids using KMeans and ensure each cluster has at least one connected road segment
    initialize = KMeans(n_clusters=num_tasks, random_state=42, n_init=1, init='random')
    initialize.fit(M)
    centroids = initialize.cluster_centers_

    # Ensure each cluster has at least one connected road segment
    labels = initialize.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    for i in range(num_tasks):
        if i not in unique_labels:
            labels[np.argmin(counts)] = i
            unique_labels, counts = np.unique(labels, return_counts=True)

    # Iterate through all states
    for episode in range(num_episodes):
        total_reward = 0

        for state in range(num_states):
            action = ac_rl_model.select_action(M[state])
            reward = -np.linalg.norm(M[state] - centroids[action])

            # Add the shortest path of the submatrix to the reward
            nodes_in_cluster = np.where(labels == action)[0]
            sub_task_matrix = M[nodes_in_cluster][:, nodes_in_cluster]

            real_ids = node_indices[nodes_in_cluster]

            for road_coord in real_ids:
                # Extract the x and y coordinates of the current road connected road segment
                x1 = road_coordinates[road_coord][0]
                y1 = road_coordinates[road_coord][1]
                # Extract the x and y coordinates of the warehouse
                x2 = warehouse_coordinate[0]
                y2 = warehouse_coordinate[1]
                warehouse_distance += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            warehouse_distance = warehouse_distance / sub_task_matrix.shape[0]

            distance_water_refill_station = [
                calculate_distance(road_coordinates[road_coord], site_coord, site_coordinates)
                for road_coord in real_ids
                for site_coord in site_coordinates]

            num_dis_water_refill_station = len(distance_water_refill_station)
            total_water_refill_station_distance = sum(distance_water_refill_station) / num_dis_water_refill_station
            min_water_refill_station_distance = min(distance_water_refill_station)

            if sub_task_matrix.shape[0] > 1:
                shortest_path_length = greedy_shortest_path(sub_task_matrix)
                shortest_path_length = shortest_path_length / sub_task_matrix.shape[0]

                # Incorporate the shortest path length, minimum distance to water refill stations,
                # and average distance to the warehouse into the reward.
                # Negative values are used as we want to minimize these distances.
                reward += -shortest_path_length
                reward += -min_water_refill_station_distance
                reward += -warehouse_distance

            total_reward += reward

            next_state = M[state + 1] if state + 1 < num_states else M[0]
            ac_rl_model.update_networks(M[state], action, reward, next_state)

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Ensure each cluster has at least three connected road segments
    unique_labels, counts = np.unique(labels, return_counts=True)
    while any(counts < 3):
        for i in range(num_tasks):
            if counts[i] < 3:
                min_count_label = unique_labels[np.argmin(counts)]
                min_count_indices = np.where(labels == min_count_label)[0]
                if len(min_count_indices) > 0:
                    labels[min_count_indices[0]] = i
        unique_labels, counts = np.unique(labels, return_counts=True)

    return labels





"""
This class implements a single USV path planning algorithm using a virtual agent-based approach inspired by Ant Colony Optimization.

It aims to find the shortest path for the USV considering factors like water consumption, travel range, and task execution time limit.

"""
class Single_USV_Path_Planning(object):
    def __init__(self, distances, n_virtual_agent, n_best, n_iterations, decay, alpha=1, beta=1,
                 initial_water_capacity=100,
                 initial_range=150, index_mapping=None, road_lengths=None, road_coordinates=None,
                 site_coordinates=None):

        self.index_mapping = index_mapping
        self.road_lengths = road_lengths
        self.road_coordinates = road_coordinates
        self.site_coordinates = site_coordinates

        self.distances = distances  # Distance matrix
        self.pheromone = np.full_like(distances, fill_value=1e-10)  # Pheromone matrix
        self.all_inds = range(len(distances))  # Indices of all connected road segments
        self.n_virtual_agent = n_virtual_agent  # Number of virtual agent
        self.n_best = n_best  # Number of n_virtual_agents that select the optimal path
        self.n_iterations = n_iterations  # Number of iterations
        self.decay = decay  # Pheromone evaporation coefficient
        self.alpha = alpha  # Importance factor of pheromone
        self.beta = beta  # Heuristic factor, indicating the expectation of virtual agents choosing the next connected road segment
        self.initial_water_capacity = initial_water_capacity  # Initial water capacity for each agent
        self.initial_range = initial_range  # Initial range for each virtual agent

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            print(f"The {i}/{self.n_iterations} generation of virtual agents begins to find a path.")
            self.water_capacity = self.initial_water_capacity  # Reset water capacity for each virtual agent
            self.range = self.initial_range  # Reset range for each virtual agent
            all_paths = self.gen_all_paths()  # Generate all paths
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)  # Diffusion pheromone
            shortest_path = min(all_paths, key=lambda x: x[1])  # Find the shortest path

            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
                # print(all_time_shortest_path)

            self.pheromone *= self.decay  # Pheromone volatilization
        if all_time_shortest_path[1] > 50000:
            all_time_shortest_path = ([], float('inf'))
        print(all_time_shortest_path)  # Return a tuple even when the path length is infinite
        return all_time_shortest_path

    # Diffusion pheromone
    def spread_pheronome(self, all_paths, n_best, shortest_path):
        # Added a very small positive number epsilon and added it to the denominator when calculating pheromone updates.
        # This way, even though self.distances[move] If  is 0, the denominator will not be 0, thus avoiding the error of dividing by 0.

        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        epsilon = 1e-10  # A small positive number to prevent division by zero
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / (self.distances[move] + epsilon)

    def gen_path_dist(self, path, non_working_distance):
        # Calculated path length
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        total_dist += non_working_distance
        return total_dist

    def gen_all_paths(self):
        # Generate all paths
        all_paths = []
        for i in range(self.n_virtual_agents):
            start = np.random.choice(self.all_inds)  # Randomly select a start connected road segment
            path, path_length = self.gen_path(start)
            all_paths.append((path, path_length))
        # try:
        #     start = np.random.choice(self.all_inds)  # Randomly select a start connected road segment
        #     path, path_length = self.gen_path(start)
        #     all_paths.append((path, path_length))
        # except Exception as e:

        #     all_paths.append(([], float('inf')))  # Add a path with infinite length
        return all_paths

    def pick_move(self, pheromone, dist, visited):
        # Select the noonnected road segmentde to move next
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        if np.any(dist == 0) or np.isnan(pheromone).any():
            # Handles division by zero and NaN values
            return random.choice(list(set(self.all_inds) - visited))

        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move

    def gen_path(self, start):
        # Generate a path
        path = []  # List to store the traversed path
        visited = set()  # Set to record visited connected road segments
        visited.add(start)  # Mark the starting virtual_agent as visited
        # Keep track of the previous virtual_agent
        prev = start

        # Initialize refill count
        refill_times = 0
        # Variable to hold the closest connected road segment
        closest_node = None
        # Variable to hold the coordinates of the closest connected road segment
        closest_coordinate = None

        # Initialize minimum refill distance
        min_water_refill_distance = 0

        # Set the task execution time limit (1.5 hours)
        task_execution_time_limit = 1.5

        # Initialize total task execution time
        task_execution_time = 0

        # Initialize total travel distance
        travel_distance = 0

        # Flag for refill status (0: no refill)
        refill = 0

        # Set travel speed to 7.5km/h
        speed = 7.5

        # Set water consumption to 15 L per km
        water_consumption_per_km = 15

        warehouse_coordinate = [-0.0, 0.0]
        x1 = self.road_coordinates[self.index_mapping[start]][0]
        y1 = self.road_coordinates[self.index_mapping[start]][1]
        x2 = warehouse_coordinate[0]
        y2 = warehouse_coordinate[1]
        warehouse_distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        non_working_distance = 0

        for i in range(len(self.distances) - 1):
            if i == 0:
                non_working_distance += warehouse_distance
                travel_distance += warehouse_distance
                # print(travel_distance)

            min_water_refill_station_distance = 0

            # Heuristic design
            Heuristic_method = self.road_lengths[self.index_mapping[prev]] + self.distances[prev] + self.distances[
                prev] / (self.road_lengths[self.index_mapping[prev]])
            move = self.pick_move(self.pheromone[prev], Heuristic_method, visited)
            path.append((prev, move))
            # Length of working section and transfer section
            Working_section = self.road_lengths[self.index_mapping[move]]
            Transfer_section = self.distances[(prev, move)]
            Total_distance_traveled = Working_section + Transfer_section

            # Update the range and water capacity of the virtual agent
            self.range -= Total_distance_traveled * 0.1 * 1e-03
            self.water_capacity -= Working_section * water_consumption_per_km

            # If the range or water capacity is less than or equal to 0, return a path with infinite length
            if self.range <= 0 or self.water_capacity <= 0:

                refill_times += 1
                # print(refill_times)
                self.range = self.initial_range
                # print(self.initial_range)
                self.water_capacity = self.initial_water_capacity

                min_water_refill_station_distance = np.inf

                target_coordinate = np.array(self.road_coordinates[self.index_mapping[prev]])
                for node, coord in self.site_coordinates.items():
                    distance = np.linalg.norm(np.array(coord) - target_coordinate)
                    if distance < min_water_refill_station_distance:
                        min_water_refill_station_distance = distance
                        closest_coordinate = coord
                        closest_node = node
            else:
                refill_times += 0
            # refill_times+=flag
            travel_distance += Total_distance_traveled + min_water_refill_station_distance
            non_working_distance += Transfer_section + min_water_refill_station_distance

            # Calculate the total task execution time. The total time consists of two parts:
            # the time spent on refilling operations and the time spent on traveling.
            # Here, we assume that each refill operation takes 0.5 hours, and the speed during travel is 15 km/h.
            # travel_distance is in meters, and we convert the speed unit (km/h) to be consistent with the distance unit (m)
            # by using 15000 (since 15 km = 15000 m).
            travel_distance_km = travel_distance / 1000
            task_execution_time = refill_times * 0.5 + travel_distance_km / speed

            if task_execution_time > task_execution_time_limit:
                # print(refill_times,task_execution_time)
                return path, float('inf')

            next_node = self.index_mapping[move]
            prev = move
            visited.add(move)
        # If the virtual agent successfully completes the path, return the path and its length
        # print(f"task_execution_time:{task_execution_time}")
        return path, self.gen_path_dist(path, non_working_distance)

def np_choice(a, size, replace=True, p=None):
    # Generates a random sample from a given one-dimensional array
    return np.random.choice(a, size, replace, p)

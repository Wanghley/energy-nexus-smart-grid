import numpy as np

class EnergyOptimizationRLAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        """
        Q-learning agent to recommend optimal hour to use electricity
        based on predicted demand, price, and carbon intensity.

        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = None
        self.num_states = None
        self.num_actions = None

    def _build_state_space(self, df):
        """
        Build a discrete state space based on input dataframe features.
        Here we discretize the continuous features into bins.

        Args:
            df: DataFrame with columns:
                'predicted_powerDemand', 'predicted_price', 'predicted_carbonIntensity'

        Returns:
            states: list of tuples representing discretized states
        """
        # Number of discrete bins per feature
        bins = 5

        power_bins = np.linspace(df['predicted_powerDemand'].min(), df['predicted_powerDemand'].max(), bins)
        price_bins = np.linspace(df['predicted_price'].min(), df['predicted_price'].max(), bins)
        carbon_bins = np.linspace(df['predicted_carbonIntensity'].min(), df['predicted_carbonIntensity'].max(), bins)

        states = []
        for _, row in df.iterrows():
            p_bin = np.digitize(row['predicted_powerDemand'], power_bins)
            pr_bin = np.digitize(row['predicted_price'], price_bins)
            c_bin = np.digitize(row['predicted_carbonIntensity'], carbon_bins)
            states.append((p_bin, pr_bin, c_bin))
        return states

    def _get_reward(self, power, price, carbon):
        """
        Reward function to maximize usage when demand is low, price low, carbon low.
        We want to *maximize* the negative of cost and carbon.

        Args:
            power, price, carbon: floats

        Returns:
            reward: float
        """
        # Example weighted sum (you can adjust weights)
        reward = - (0.5 * power + 0.3 * price + 0.2 * carbon)
        return reward

    def train(self, df, episodes=500):
        """
        Train the Q-learning agent on the given forecast dataframe.

        Args:
            df: DataFrame with predicted_powerDemand, predicted_price, predicted_carbonIntensity columns
            episodes: Number of training iterations
        """
        states = self._build_state_space(df)
        self.num_states = len(states)
        self.num_actions = len(states)  # actions correspond to choosing a specific hour index

        # Initialize Q-table with zeros
        self.q_table = np.zeros((self.num_states, self.num_actions))

        for ep in range(episodes):
            # Start from a random state
            state_idx = np.random.randint(0, self.num_states)

            done = False
            while not done:
                # Choose action (hour index) using epsilon-greedy policy
                if np.random.rand() < self.epsilon:
                    action_idx = np.random.randint(0, self.num_actions)
                else:
                    action_idx = np.argmax(self.q_table[state_idx])

                # Get reward for action (using row corresponding to action)
                power = df.iloc[action_idx]['predicted_powerDemand']
                price = df.iloc[action_idx]['predicted_price']
                carbon = df.iloc[action_idx]['predicted_carbonIntensity']
                reward = self._get_reward(power, price, carbon)

                # Next state is action's state
                next_state_idx = action_idx

                # Update Q-table
                best_next_action = np.argmax(self.q_table[next_state_idx])
                td_target = reward + self.gamma * self.q_table[next_state_idx][best_next_action]
                td_error = td_target - self.q_table[state_idx][action_idx]

                self.q_table[state_idx][action_idx] += self.alpha * td_error

                # Episode ends after one step (one decision)
                done = True

    def recommend_hour(self, df):
        """
        Recommend the best hour index to use electricity from the forecast dataframe.

        Args:
            df: DataFrame with predicted_powerDemand, predicted_price, predicted_carbonIntensity columns

        Returns:
            recommended_hour_index: int
        """
        if self.q_table is None:
            raise ValueError("RL Agent not trained yet")

        states = self._build_state_space(df)
        current_state_idx = 0  # We just use the first state or dummy state here

        # Choose the best action from Q-table for current state
        best_action = np.argmax(self.q_table[current_state_idx])

        return best_action

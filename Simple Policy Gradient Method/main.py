import numpy as np 

class SimpleGridWorld:
    def __init__(self, grid_size, start, goal):
        self.grid_size = grid_size 
        self.goal = goal 
        self.start = start 
        self.state = start 
    
    def reset(self):
        self.state = self.start 
        return self.state 
    
    def step(self, action):
        x, y= self.state 
        if action == 0 : 
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.grid_size - 1, y + 1)
        self.state = (x, y)
        reward = - 1 if self.state != self.goal else 0 
        done= self.state == self.goal 
        return self.state, reward, done 
    
    def get_possible_actions(self):
        return [0, 1, 2, 3]
    

class PolicyGradientAgent:
    def __init__(self, env, learning_rate = 0.01, gamma = 0.99):
        self.env = env 
        self.learning_rate = learning_rate 
        self.gamma= gamma 
        self.policy = np.random.rand(env.grid_size, env.grid_size, 4)
        self.policy = np.exp(self.policy) / np.exp(self.policy).sum(axis = 2, keepdims = True)
    
    def choose_actions(self, state):
        x, y = state 
        return np.random.choice(4, p = self.policy[x,y])
    
    def train(self, episodes):
        for episode in range(episodes):
            states, actions, rewards = [], [], []
            state = self.env.reset() 
            while True: 
                action = self.choose_actions(state)
                next_state, reward, done = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state =next_state 
                if done:
                    break 
            self.update_policy(states, actions, rewards) 
        
    def update_policy(self, states, actions, rewards):
        discounted_rewards = self.discount_rewards(rewards)
        for i,(state, action) in enumerate(zip(states, actions)):
            x, y = state 
            self.policy[x, y, action] += self.learning_rate * discounted_rewards[i]
            self.policy[x,y] = np.exp(self.policy[x,y]) / np.exp(self.policy[x,y]).sum()
        
    def discount_rewards(self, rewards):
        discounted = np.zeros_like(rewards, dtype =np.float32)
        cumulative = 0 
        for i in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma  + rewards[i]
            discounted[i] = cumulative 
        return discounted 


if __name__ == '__main__':
    grid_size = 5
    start = (0, 0)
    goal = (4,4)
    env = SimpleGridWorld(grid_size, start, goal)
    agent = PolicyGradientAgent(env)
    agent.train(episodes = 500)
    for x in  range(grid_size):
        for y in range(grid_size):
            action = np.argmax(agent.policy[x,y])
            print(f"State: ({x}, {y}), Action : {['Up', 'Down', 'Left', 'Right']}")
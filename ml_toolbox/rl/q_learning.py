
from environment import MountainCar
import sys
import numpy as np


def write_rewards(rewards, output_filename):
    with open(output_filename, 'w') as f:
        for r in rewards:
            print(r, file = f)
    
def write_weights(weights, bias, output_filename):
    with open(output_filename, 'w') as f:
        print(bias, file = f)
        
        for row in range(weights.shape[0]):
            for col in range(weights.shape[1]):
                print(weights[row][col], file =f)

def return_q(q_weights: np.ndarray, states: dict, action: int, bias: float):  
    #q_val = np.array(state.values()) @ q_weights[:, action]
    
    q_vals = [float(state_val) * float(q_weights[int(state_idx), action]) \
                for state_idx, state_val in states.items()] 

    return (sum(q_vals) + float(bias))
        
    
def return_max_q(q_weights: np.ndarray, states: dict, bias : float):
    
    q_vals = [return_q(q_weights, states, i, bias) for i in range(q_weights.shape[1])] 
    
    return np.max(q_vals), np.argmax(q_vals)


def select_action(q_weights: int, states:dict, epsilon: float, bias : float) -> int:
    if np.random.random() <= epsilon:
        return np.random.choice(q_weights.shape[1])
    else:
        return return_max_q(q_weights, states, bias)[1]

def moving_average(a, n=25) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n   

def main(args):
    
    mode = str(args[1])
    output_weights_file = str(args[2])
    output_returns = str(args[3])
    episodes = int(args[4])
    max_iterations = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    learning_rate = float(args[8])
    num_actions = 3
    agent = MountainCar(mode)
    q_weights = np.zeros([agent.state_space, 3], dtype=np.longdouble)
    bias = 0.0

    rewards = [0] * episodes

    for episode in range(episodes):
        state = agent.reset()
        
        for iters in range(max_iterations):
        
            action = select_action(q_weights, state, epsilon, bias)
            
            q_cur = return_q(q_weights, state, action, bias)
            next_state, reward, done = agent.step(action)
            
            rewards[episode] += reward
            
            q_star = reward + gamma * return_max_q(q_weights, next_state, bias)[0]

            delta_l =  learning_rate * (q_cur - q_star)

            
            for state_idx, state_val in state.items():
                q_weights[state_idx, action] -=  state_val * delta_l

            bias -= delta_l

            state = next_state
               
            if done == True:
                break

    
    write_rewards(rewards, output_returns)
    write_weights(q_weights, bias, output_weights_file)

    rewards = np.array(rewards)
    np.savez(f'rewards_{mode}.npz', rewards = np.array(rewards))    


if __name__ == "__main__":
    main(sys.argv)



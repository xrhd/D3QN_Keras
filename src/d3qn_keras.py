from keras.layers import Dense, Activation, Input, Lambda
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam
from keras_radam import RAdam
import keras.backend as K
import numpy as np
from src.per import ReplayBuffer, PER


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    inp = Input((input_dims,))
    x = Dense(fc1_dims, activation='relu')(inp)
    x = Dense(fc2_dims, activation='relu')(x)

    A = Dense(n_actions)(x)
    V = Dense(1)(x)
    
    def q_values(inp):
        V, A = inp[0], inp[1]
        return V + A - K.mean(A, keepdims=True)
    Q = Lambda(q_values)([V,A])

    model = Model(inp, Q)
    model.compile(optimizer=RAdam(lr=lr), loss='mse')
    return model
    

class D3QNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, 
                 epsilon_dec=0.996,  epsilon_end=0.01, build_dqn=build_dqn,
                 mem_size=1000000, fname='d3qn_model_radam.h5', replace_target=100):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = PER(mem_size, input_dims, n_actions, discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)
        self.q_target = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        trasition = state, action, reward, new_state, done
        self.memory.store_transition(trasition)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            self._decrise_epsilon()
            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file, custom_objects={'RAdam': RAdam})
        # if we are in evaluation mode we want to use the best weights for
        # q_target
        if self.epsilon == 0.0:
            self.update_network_parameters()

    def _decrise_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                           self.epsilon_min else self.epsilon_min

if __name__ == "__main__":
    agent = D3QNAgent(alpha=0.0005, gamma=0.99, n_actions=4, epsilon=0.0,
                      batch_size=64, input_dims=8)
    print(agent.q_target.summary())
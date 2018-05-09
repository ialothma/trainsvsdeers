import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
#import tflearn

MAX_DEER = 200

class dvt():

    game_status = False

    def __init__(self, difficulty = 20, X = 1280, Y = 160):
        self.X_AXIS = X
        self.Y_AXIS = Y
        self.total_deer_number = difficulty
        self.generate_deer()
        self.generate_train()
        self.reward = 0
        self.n_actions = 5
        self.state_size = 3*MAX_DEER + 3
        self.state = []
        self.deer_state = []
        self.train_state = []
        self.done = None
        self.steps_beyond_done = None

    def generate_deer(self):

        deer_key = ['deer_number', 'deer_velocity', 'deer_coordinates_x', 'deer_coordinates_y']
        deers = []
        n = 0
        for n in range(self.total_deer_number):
            #print(n)
            deerz = [n, 1, (random.randint(0, self.X_AXIS)), (random.randint(1, self.Y_AXIS))]
            deers.append(dict(zip(deer_key, deerz)))
        self.deers = deers

    def generate_train(self):

        train = {'train_velocity': (random.randint(2, 10)),
                 'train_coordinates_x': (random.randint(0, self.X_AXIS)),
                 'train_coordinates_y': 0}
        self.train = train

    def train_move(self):

        if  self.train['train_velocity'] > 0:
            self.train['train_coordinates_y'] = self.train['train_coordinates_y'] + self.train['train_velocity']



    def deer_move(self):
        movement_types=['fwd','fwd_r','fwd_l','bwd','bwd_r','bwd_l','left','right','stay']
        for m in range(self.total_deer_number):
            movement_decision = random.choice(movement_types)
            #print('#########################################')
            #print(movement_decision)
            #print(m)
            if movement_decision == 'fwd':
                self.deers[m]['deer_coordinates_y'] = self.deers[m]['deer_coordinates_y'] + self.deers[m]['deer_velocity']
            if movement_decision == 'fwd_r':
                self.deers[m]['deer_coordinates_y'] = self.deers[m]['deer_coordinates_y'] + self.deers[m]['deer_velocity']
                self.deers[m]['deer_coordinates_x'] = self.deers[m]['deer_coordinates_x'] + self.deers[m]['deer_velocity']
            if movement_decision == 'fwd_l':
                self.deers[m]['deer_coordinates_y'] = self.deers[m]['deer_coordinates_y'] + self.deers[m]['deer_velocity']
                self.deers[m]['deer_coordinates_x'] = self.deers[m]['deer_coordinates_x'] - self.deers[m]['deer_velocity']
            if movement_decision == 'bwd':
                self.deers[m]['deer_coordinates_y'] = self.deers[m]['deer_coordinates_y'] - self.deers[m]['deer_velocity']
            if movement_decision == 'bwd_r':
                self.deers[m]['deer_coordinates_y'] = self.deers[m]['deer_coordinates_y'] - self.deers[m]['deer_velocity']
                self.deers[m]['deer_coordinates_x'] = self.deers[m]['deer_coordinates_x'] + self.deers[m]['deer_velocity']
            if movement_decision == 'bwd_l':
                self.deers[m]['deer_coordinates_y'] = self.deers[m]['deer_coordinates_y'] - self.deers[m]['deer_velocity']
                self.deers[m]['deer_coordinates_x'] = self.deers[m]['deer_coordinates_x'] - self.deers[m]['deer_velocity']
            if movement_decision == 'right':
                self.deers[m]['deer_coordinates_x'] = self.deers[m]['deer_coordinates_x'] + self.deers[m]['deer_velocity']
            if movement_decision == 'left':
                self.deers[m]['deer_coordinates_x'] = self.deers[m]['deer_coordinates_x'] - self.deers[m]['deer_velocity']
        #print("###############NEW POSITIONING###########")
        #print(deers)
        #return deers

    def extract_deer_state(self):

        self.deer_state=[]

        for deer_x in self.deers:
            b={'deer_velocity':deer_x['deer_velocity'],
               'deer_coordinates_x':deer_x['deer_coordinates_x'],
               'deer_coordinates_y':deer_x['deer_coordinates_y']}

            self.deer_state.append(b.values())

        #print self.deer_state

    def extract_train_state(self):

        self.train_state=[self.train['train_velocity'],
                          self.train['train_coordinates_x'],
                          self.train['train_coordinates_y']]

        #print self.train_state

    def set_speed(self, choice):
        acc = [-2, -1, 0, 1, 2]
        #print ("self.train type= %s", type(self.train['train_velocity']))
        #print ("Choice type= %s", type(choice))
        if self.train['train_velocity'] + acc[choice] in range(0,11):
            self.train['train_velocity'] = self.train['train_velocity'] + acc[choice]

    def realtime_check(self):

        #ml_acceleration_value=ml_speed_decide(extract_deer_position(deers),extract_train_position(train))

        #train['train_velocity']=train['train_velocity']+ml_acceleration_value

        dangerzone = self.train["train_coordinates_y"] + self.train["train_velocity"]


        for deer_x in self.deers:
            if deer_x['deer_coordinates_x'] == self.train['train_coordinates_x'] \
            and deer_x['deer_coordinates_y'] in range(self.train['train_coordinates_y'], dangerzone):
               print("Deer Position: %s, Train Position: %s"%( deer_x['deer_coordinates_y'],self.train['train_coordinates_y']))
               return True
        return False


    def step(self ,action):

        #print(train)
        #print(deers)
        self.set_speed(action)
        self.deer_move()
        self.train_move()
        print(self.train['train_coordinates_y'],self.train['train_coordinates_x'])
        self.reward, self.done = self.game_result()
        self.extract_train_state()
        self.extract_deer_state()
        self.state = self.deer_state
        self.state.append(self.train_state)

        d_x, d_y = [], []

        for deer_tmp in self.deers:
            d_x.append(deer_tmp['deer_coordinates_x'])
            d_y.append(deer_tmp['deer_coordinates_y'])


        self.deer_loc.set_xdata(d_x)
        self.deer_loc.set_ydata(d_y)
        self.train_loc.set_ydata(self.train['train_coordinates_y'])
        plt.pause(0.1)

        return np.array(self.state), self.reward, self.done, {}

    def game_result(self):

        if self.realtime_check():
            self.reward= 1.0
            self.game_status= True
            print("GAME WILL BE RESTARTED THERE IS A CRASH")
        elif self.train['train_velocity'] == 0:
            self.reward = 0
            self.game_status = False
        elif self.train["train_coordinates_y"] >= self.Y_AXIS:
            self.game_status = True
            self.reward= 1.0
            print("GAME WILL BE RESTARTED THERE IS NO CRASH")

        return self.reward, self.game_status

    def reset(self):
        #print(self.total_deer_number)
        self.generate_deer()
        self.generate_train()
        self.reward = 0
        self.game_status = False
        #print(self.train['train_coordinates_y'],self.train['train_coordinates_x'])
        self.extract_train_state()
        self.extract_deer_state()

        self.state = self.deer_state
        self.state.append(self.train_state)

        return np.array(self.state)

    def map(self):

        d_x, d_y = [], []

        for deer_tmp in self.deers:
            d_x.append(deer_tmp['deer_coordinates_x'])
            d_y.append(deer_tmp['deer_coordinates_y'])



        ax = fig.add_subplot(111)
        self.deer_loc, = ax.plot(d_x, d_y, 'rs', markersize=4)
        self.train_loc, = ax.plot(self.train['train_coordinates_x'], self.train['train_coordinates_y'], 'g^', markersize=8)

        fig.canvas.draw()
        plt.pause(0.1)

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
    env = dvt(MAX_DEER)
    state_size = env.state_size
    action_size = env.n_actions
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    #plt.ion()
    fig = plt.figure()
    plt.axis([0,1280,0,160])
    plt.grid()
    env.map()

    for e in range(EPISODES):
        plt.title("Episode %s"%e)
        state = env.reset()
        print (state)
        state = np.reshape(state, [1, state_size])

        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)


# if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
"""
def env():

    total_reward = 0
    game = dvt(2000)
    while True:


        game.game_move()

        if game.train['train_coordinates_y'] >= 160 or game.game_status == False:
            print("end of round")
            game.game_result()
            total_reward = game.reward + total_reward
            game.reset_game()
        #game.game_result()
        deer_data_set = game.extract_deer_position
        train_data_set = game.extract_train_position

        #print("acceleration?")

        #game.set_speed(int(raw_input() or "0"))

        print(total_reward)
        #print(train)
        if total_reward == 100:
            break
    #print('Final Reward:', total_reward)
    #print(train['train_coordinates_y'])
    return total_reward

if __name__ == '__main__':
    main()
"""

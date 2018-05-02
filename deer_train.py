import random


class dvt():

    total_deer_number = 20
    game_status = True
    def __init__(self, difficulty):
        self.generate_deer()
        self.generate_train()
        self.reward = 0
        total_deer_number = difficulty

    def generate_deer(self):

        deer_key = ['deer_number', 'deer_velocity', 'deer_coordinates_x', 'deer_coordinates_y']
        deers = []
        n = 0
        for n in range(self.total_deer_number):
            deerz = [n, 1, (random.randint(0, 1280)), (random.randint(1, 160))]
            deers.append(dict(zip(deer_key, deerz)))
        self.deers = deers

    def generate_train(self):

        train = {'train_number': 1, 'train_velocity': (random.randint(2, 10)),
                 'train_coordinates_x': (random.randint(0, 1280)), 'train_coordinates_y': 0}
        self.train = train

    def train_move(self):

        self.train['train_coordinates_y'] = self.train['train_coordinates_y'] + self.train['train_velocity']



    def deer_move(self):
        movement_types=['fwd','fwd_r','fwd_l','bwd','bwd_r','bwd_l','left','right','stay']
        for m in range(0,self.total_deer_number):
            movement_decision = random.choice(movement_types)
            #print('#########################################')
            #print(movement_decision)
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

    def extract_deer_position(self):

        deer_positions=[]

        for deer_x in self.deers:
            b={'deer_coordinates_x':deer_x['deer_coordinates_x'],'deer_coordinates_y':deer_x['deer_coordinates_y']}
            deer_positions.append(dict(b))

        return deer_positions


    def extract_train_position(self):

        train_position={'train_coordinates_x':self.train['train_coordinates_x'],
                        'train_coordinates_y':self.train['train_coordinates_y']}

        return train_position

    def set_speed(self, choice):
        self.train['train_velocity'] = self.train['train_velocity'] + choice

    def realtime_check(self):

        #ml_acceleration_value=ml_speed_decide(extract_deer_position(deers),extract_train_position(train))

        #train['train_velocity']=train['train_velocity']+ml_acceleration_value

        for deer_x in self.deers:
            y=0
            if deer_x['deer_coordinates_x'] == self.train['train_coordinates_x']:
                while y <= self.train['train_velocity']:
                    if deer_x['deer_coordinates_y'] == self.train['train_coordinates_y'] + y:
                       print(deer_x['deer_coordinates_y'],self.train['train_coordinates_y'])
                       return False
                    y+=1
        return True


    def game_move(self):

        #print(train)
        #print(deers)
        if self.realtime_check() == False:
            self.game_status = False
        else:
            self.deer_move()
            self.train_move()
            print(self.train['train_coordinates_y'],self.train['train_coordinates_x'])
            self.game_status = True


    def game_result(self):

        if self.game_status == False:
            self.reward=self.reward - 10
            print("GAME WILL BE RESTARTED THERE IS A CRASH")
        elif self.game.train['train_coordinates_y'] >= 160:
            self.reward=self.reward + 10
            print("GAME WILL BE RESTARTED THERE IS NO CRASH")
            #print("Current Reward is:",reward)

    def reset_game(self):
        self.generate_deer
        self.generate_train

def main():

    total_reward = 0
    game = dvt(2000)
    while True:

        if game.train['train_coordinates_y'] >= 160:
            game.reset_game()

        game.game_move()
        game.game_result()
        game.set_speed(ml_output)
        total_reward = game.reward + total_reward
        print(total_reward)
        #print(train)
        if total_reward == 100:
            break
    #print('Final Reward:', total_reward)
    #print(train['train_coordinates_y'])
    return total_reward

if __name__ == '__main__':
    main()

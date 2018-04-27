import random

total_deer_number=1
init_train={}
init_deers=[]

def generate_deer():

    deer_key = ['deer_number', 'deer_velocity', 'deer_coordinates_x', 'deer_coordinates_y']
    deers = []
    n = 0
    while n < total_deer_number:
        deerz = [n, 1, (random.randint(0, 1280)), (random.randint(1, 160))]
        deers.append(dict(zip(deer_key, deerz)))
        n = n + 1
    return deers

def generate_train():

    train = {'train_number': 1, 'train_velocity': (random.randint(2, 10)),
             'train_coordinates_x': (random.randint(0, 1280)), 'train_coordinates_y': 0}
    return train

def train_move(train):

    train['train_coordinates_y'] = train['train_coordinates_y'] + train['train_velocity']

    return train


def deer_move(deers):
    movement_types=['fwd','fwd_r','fwd_l','bwd','bwd_r','bwd_l','left','right','stay']
    for m in range(0,total_deer_number):
        movement_decision = random.choice(movement_types)
        #print('#########################################')
        #print(movement_decision)
        if movement_decision == 'fwd':
            deers[m]['deer_coordinates_y'] = deers[m]['deer_coordinates_y'] + deers[m]['deer_velocity']
        if movement_decision == 'fwd_r':
            deers[m]['deer_coordinates_y'] = deers[m]['deer_coordinates_y'] + deers[m]['deer_velocity']
            deers[m]['deer_coordinates_x'] = deers[m]['deer_coordinates_x'] + deers[m]['deer_velocity']
        if movement_decision == 'fwd_l':
            deers[m]['deer_coordinates_y'] = deers[m]['deer_coordinates_y'] + deers[m]['deer_velocity']
            deers[m]['deer_coordinates_x'] = deers[m]['deer_coordinates_x'] - deers[m]['deer_velocity']
        if movement_decision == 'bwd':
            deers[m]['deer_coordinates_y'] = deers[m]['deer_coordinates_y'] - deers[m]['deer_velocity']
        if movement_decision == 'bwd_r':
            deers[m]['deer_coordinates_y'] = deers[m]['deer_coordinates_y'] - deers[m]['deer_velocity']
            deers[m]['deer_coordinates_x'] = deers[m]['deer_coordinates_x'] + deers[m]['deer_velocity']
        if movement_decision == 'bwd_l':
            deers[m]['deer_coordinates_y'] = deers[m]['deer_coordinates_y'] - deers[m]['deer_velocity']
            deers[m]['deer_coordinates_x'] = deers[m]['deer_coordinates_x'] - deers[m]['deer_velocity']
        if movement_decision == 'right':
            deers[m]['deer_coordinates_x'] = deers[m]['deer_coordinates_x'] + deers[m]['deer_velocity']
        if movement_decision == 'left':
            deers[m]['deer_coordinates_x'] = deers[m]['deer_coordinates_x'] - deers[m]['deer_velocity']
        m=m+1
    #print("###############NEW POSITIONING###########")
    #print(deers)
    return deers

def extract_deer_position(deers):

    deer_positions=[]

    for deer_x in deers:
        b={'deer_coordinates_x':deer_x['deer_coordinates_x'],'deer_coordinates_y':deer_x['deer_coordinates_y']}
        deer_positions.append(dict(b))

    return deer_positions


def extract_train_position(train):

    train_position={'train_coordinates_x':train['train_coordinates_x'],'train_coordinates_y':train['train_coordinates_y']}

    return train_position

def ml_speed_decide(deer_position,train_position):

    ml_influence_range = [-5,-4,-3,-2,-1,0,1,2,3,4,5]

    return ml_influence

def realtime_check(deers,train):

    ml_acceleration_value=ml_speed_decide(extract_deer_position(deers),extract_train_position(train))
    
    train['train_velocity']=train['train_velocity']+ml_acceleration_value

    for deer_x in deers:
        y=0
        if deer_x['deer_coordinates_x'] == train['train_coordinates_x']:
            while y <= train['train_velocity']:
                if deer_x['deer_coordinates_y'] == train['train_coordinates_y'] + y:
                   print(deer_x['deer_coordinates_y'],train['train_coordinates_y'])
                   return False
                y+=1
    return True


def game_move(deers,train):

    #print(train)
    #print(deers)
    while train['train_coordinates_y'] <= 160:
        if realtime_check(deers,train) == False:
            return False
        else:
            deer_move(deers)
            train_move(train)
            print(train['train_coordinates_y'],train['train_coordinates_x'])
    return True


def start_game(deers,train):

    reward=0

    if game_move(deers,train) == False:
        reward=reward-10
        #print("GAME WILL BE RESTARTED THERE IS A CRASH")
    else:
        reward=reward + 10
        #print("GAME WILL BE RESTARTED THERE IS NO CRASH")
    #print("Current Reward is:",reward)

    return reward

def main():

    total_reward = 0
    while True:
        deers = generate_deer()
        train = generate_train()
        total_reward = start_game(deers,train)+total_reward
        print(total_reward)
        #print(train)
        if total_reward == 100:
            break
    #print('Final Reward:', total_reward)
    #print(train['train_coordinates_y'])
    return total_reward

if __name__ == '__main__':
    main()



import keras.backend as K
import numpy as np
from keras.layers import Dense, Input, LeakyReLU, Subtract
from keras.models import Model
from keras.optimizers import Adam

# from utils import gen_sample, flatten_1d_b, inv_action_map, perc_solved_cube
import keras.backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm

from collections import Counter
def perc_solved_cube(cube):
    flat = cube.current_state.reshape(-1)
    perc_side = [order(flat[i:(i + 4)]) for i in range(0, 4 * 6, 4)]
    return np.mean(perc_side)
def order(data):
    if len(data) <= 1:
        return 0

    counts = Counter()

    for d in data:
        counts[d] += 1

    probs = [float(c) / len(data) for c in counts.values()]

    return max(probs)


def acc(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())


def get_model(lr=0.0001):
    input1 = Input((288*2,))

    d1 = Dense(1024)
    d2 = Dense(1024)
    d3 = Dense(1024)

    d4 = Dense(50)

    x1 = d1(input1)
    x1 = LeakyReLU()(x1)
    x1 = d2(x1)
    x1 = LeakyReLU()(x1)
    x1 = d3(x1)
    x1 = LeakyReLU()(x1)
    x1 = d4(x1)
    x1 = LeakyReLU()(x1)

    out_value = Dense(1, activation="linear", name="value")(x1)
    out_policy = Dense(len(action_map), activation="softmax", name="policy")(x1)

    model = Model(input1, [out_value, out_policy])

    model.compile(loss={"value": "mae", "policy": "sparse_categorical_crossentropy"}, optimizer=Adam(lr),
                  metrics={"policy": acc})
    model.summary()

    return model

action_map = {"F" : 0, "B" : 1, "L" : 2, "R" : 3, "U" : 4, "D" : 5, "Fprime" :6 , "Bprime" : 7,
                   "Lprime" : 8, "Rprime" : 9, "Uprime" : 10, "Dprime" :11}
import pickle
dbn = pickle.load(open("/content/dbn_model00004_0634_576.sav", 'rb'))

file_path = "/content/auto288_576.h5"
from cube_solver_v3 import *
import copy
model = get_model()

model.load_weights(file_path)

transformation_moves, transformation=generate_sequence(10)
print(transformation_moves)
cube = rubix2x2()
cube.list_execution(transformation_moves)
cube.score = 0
list_sequences = [[cube]]

existing_cubes = set()
cube.history = []

for j in range(15):
    X=[]
    for x in list_sequences:
        x[-1].convert_to_binary()
        X.append(x[-1].binary_state)

    value, policy = model.predict(dbn.transform(np.array(X)), batch_size=1024)
    new_list_sequences = []

    for x, policy in zip(list_sequences, policy):
        new_sequences = []
        for action in action_map:
            temp = copy.deepcopy(x[-1])
            temp.list_execution([action])
            new_sequences.append(x + [temp])
            
        pred = np.argsort(policy)
        cube_1 = copy.deepcopy(x[-1])
        cube_1.list_execution([list(action_map.keys())[pred[-1]]])

        cube_2 = copy.deepcopy(x[-1])
        cube_2.list_execution([list(action_map.keys())[pred[-2]]])

        new_list_sequences.append(x + [cube_1])
        new_list_sequences.append(x + [cube_2])

    last_states_flat=[]
    for k in list_sequences:
        k[-1].convert_to_binary()
        last_states_flat.append(k[-1].binary_state)
    value, _ = model.predict(dbn.transform(np.array(last_states_flat)), batch_size=1024)
    value = value.ravel().tolist()
    for x, v in zip(new_list_sequences, value):
        x[-1].score = v if str(x[-1]) not in existing_cubes else -1

    new_list_sequences.sort(key=lambda x: x[-1].score , reverse=True)
    new_list_sequences = new_list_sequences[:100]
    existing_cubes.update(set([str(x[-1]) for x in new_list_sequences]))
    list_sequences = new_list_sequences
    list_sequences.sort(key=lambda x: perc_solved_cube(x[-1]), reverse=True)

    prec = perc_solved_cube((list_sequences[0][-1]))
    print(prec)
    if prec == 1:
        break

print(perc_solved_cube(list_sequences[0][-1]))
history =list_sequences[0][-1].history
print(history)


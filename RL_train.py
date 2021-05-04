import pickle
import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm

from cube_solver_v3 import generate_sequence, rubix2x2
import copy
from collections import Counter
from random import choice
import multiprocessing


dbn = pickle.load(open("dbn_model00004_0636.sav", 'rb'))
labels = {"F" : 0, "B" : 1, "L" : 2, "R" : 3, "U" : 4, "D" : 5, "Fprime" :6 , "Bprime" : 7,
				   "Lprime" : 8, "Rprime" : 9, "Uprime" : 10, "Dprime" :11}





def acc(y_true, y_pred):
	return K.cast(K.equal(K.max(y_true, axis=-1),
						  K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
				  K.floatx())


def get_model(lr=0.0001):
	input1 = Input((576,))

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
	out_policy = Dense(len(labels), activation="softmax", name="policy")(x1)

	model = Model(input1, [out_value, out_policy])

	model.compile(loss={"value": "mae", "policy": "sparse_categorical_crossentropy"}, optimizer=Adam(lr),
				  metrics={"policy": acc})
	model.summary()

	return model


def get_cubes_and_moves(seq_len= 15):
	transformation_moves, transformation = generate_sequence(seq_len)	
	cubes_s = []
	distange_to_solved = list(range(seq_len,0,-1))
	for i in range(seq_len):
		cube = rubix2x2()
		cube.list_execution(transformation_moves[:i+1])
		cubes_s.append(cube)
	return cubes_s[::-1], distange_to_solved

def get_all_possible_actions_cube_small(cube):

	flat_cubes = []
	rewards = []
	for a in labels.keys():
		cube_copy = copy.deepcopy(cube)
		cube_copy.list_execution([a])
		# print(cube_copy.current_state)
		cube_copy.convert_to_binary()

		flat_cubes.append(cube_copy.binary_state)
		rewards.append(2*int(perc_solved_cube(cube_copy)>0.99)-1)
	return flat_cubes, rewards

def order(data):
	if len(data) <= 1:
		return 0

	counts = Counter()

	for d in data:
		counts[d] += 1

	probs = [float(c) / len(data) for c in counts.values()]

	return max(probs)


def perc_solved_cube(cube):
	flat = cube.current_state.reshape(-1)
	perc_side = [order(flat[i:(i + 4)]) for i in range(0, 4 * 6, 4)]
	return np.mean(perc_side)

def chunker(seq, size):
	return (seq[pos:pos + size] for pos in range(0, len(seq), size))



if __name__ == "__main__":

	pool = multiprocessing.Pool(4)

	N_SAMPLES = 100
	N_EPOCH = 100

	file_path = "auto.h5"

	checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

	early = EarlyStopping(monitor="val_loss", mode="min", patience=1000)

	reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.1, patience=50, min_lr=1e-8)

	callbacks_list = [checkpoint, early, reduce_on_plateau]

	model = get_model(lr=0.0001)

	for i in tqdm(range(N_EPOCH)):
		cubes = []
		distance_to_solved = []
		for j in range(N_SAMPLES):
			_cubes, _distance_to_solved = get_cubes_and_moves(15)
			cubes.extend(_cubes)
			distance_to_solved.extend(_distance_to_solved) 

		cube_next_reward = []
		flat_next_states = []
		cube_flat = []
		for c in cubes:

			flat_cubes, rewards = get_all_possible_actions_cube_small(c)
			cube_next_reward.append(rewards)
			flat_next_states.extend(flat_cubes)
			c.convert_to_binary()
			cube_flat.append(c.binary_state)

		for _ in tqdm(range(20)):

			cube_target_value = []
			cube_target_policy = []

			next_state_value, _ = model.predict(dbn.transform(np.array(flat_next_states)), batch_size=1024)
			next_state_value = next_state_value.ravel().tolist()
			next_state_value = list(chunker(next_state_value, size=len(labels)))

			for c, rewards, values in zip(cubes, cube_next_reward, next_state_value):
				r_plus_v = 0.4*np.array(rewards) + np.array(values)
				target_v = np.max(r_plus_v)
				target_p = np.argmax(r_plus_v)
				cube_target_value.append(target_v)
				cube_target_policy.append(target_p)

			cube_target_value = (cube_target_value-np.mean(cube_target_value))/(np.std(cube_target_value)+0.01)

			# print(cube_target_policy[-30:])
			# print(cube_target_value[-30:])

			sample_weights = 1. / np.array(distance_to_solved)
			sample_weights = sample_weights * sample_weights.size / np.sum(sample_weights)

			model.fit(dbn.transform(np.array(cube_flat)), [np.array(cube_target_value), np.array(cube_target_policy)[..., np.newaxis]],
					  epochs=1, batch_size=128, sample_weight=[sample_weights, sample_weights])
			# sample_weight=[sample_weights, sample_weights],

		model.save_weights(file_path)



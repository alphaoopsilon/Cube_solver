
from cube_solver_v3 import rubix2x2, generate_sequence
import numpy as np
from tqdm import tqdm
from models import UnsupervisedDBN
import pickle

def generate_dataset(num_trials = 2000, seq_len =15, cube_size = 2):
	labels = np.zeros((num_trials, seq_len))
	cubes = np.zeros((num_trials, seq_len ,6*6*cube_size*cube_size))

	for i in tqdm(range(num_trials)):
		cube = rubix2x2()
		seq, seq_no = generate_sequence(seq_len)
		for j, move in enumerate(seq):
			cube.list_execution([move])
			cube.convert_to_binary()
			cubes[i,j,:]=cube.binary_state
		labels[i, :] = seq_no

	return cubes, labels



dataset, target = generate_dataset()




dbn = UnsupervisedDBN(hidden_layers_structure=[150, 100],
                      batch_size=100,
                      learning_rate_rbm=0.06,
                      n_epochs_rbm=8,
                      activation_function='sigmoid')
dbn.fit(dataset.reshape(-1,144))


pickle.dump(dbn, open("dbn_model.sav", 'wb'))



# dbn = UnsupervisedDBN(hidden_layers_structure=[150, 100],
#                       batch_size=10,
#                       learning_rate_rbm=0.06,
#                       n_epochs_rbm=20,
#                       activation_function='sigmoid')
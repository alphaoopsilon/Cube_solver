import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import sys
np.set_printoptions(threshold=sys.maxsize)
from random import choice
import random


"""
White 0
Green 1
Red 2
Blue 3
Orange 4
Yellow 5

The moves are performed with yellow on top and facing the green side!

"""

class rubix2x2:
	def __init__(self):
		# print("W0, G1, R2, B3, O4, Y5")
		self.methods = {"F" : self.F, "B" : self.B, "L" : self.L, "R" : self.R, "U" : self.U, "D" : self.D, "F2" : self.F2, "B2" : self.B2,
		           "L2" : self.L2, "R2" : self.R2, "U2" : self.U2, "D2" : self.D2, "Fprime" : self.Fprime, "Bprime" : self.Bprime,
		            "Lprime" : self.Lprime, "Rprime" : self.Rprime, "Uprime" : self.Uprime, "Dprime" : self.Dprime}

		self.current_state = np.array([[[0,0],[0,0]],
							  [[1,1],[1,1]],
							  [[2,2],[2,2]],
							  [[3,3],[3,3]],
							  [[4,4],[4,4]],
							  [[5,5],[5,5]]],dtype = np.float32)
		self.final_state = np.array([[[0,0],[0,0]],
							  [[1,1],[1,1]],
							  [[2,2],[2,2]],
							  [[3,3],[3,3]],
							  [[4,4],[4,4]],
							  [[5,5],[5,5]]],dtype = np.float32)
		self.binary_state = None

		self.labels = {"F" : 0, "B" : 1, "L" : 2, "R" : 3, "U" : 4, "D" : 5, "Fprime" :6 , "Bprime" : 7,
		           "Lprime" : 8, "Rprime" : 9, "Uprime" : 10, "Dprime" :11}

		self.reverse_labels = { 0: "F" ,  1: "B" ,  2: "L" ,  3: "R" ,  4: "U" ,  5: "D" ,  6: "Fprime" ,  7: "Bprime" ,
		            8: "Lprime" ,  9: "Rprime" ,  10: "Uprime" ,  11: "Dprime"}
		self.colour_mapping = {0: "W",1:"G",2:"R",3:"B",4:"O",5:"Y"}

		self.history = []

	def print_state(self):
		print(f"      [{self.colour_mapping[int(self.current_state[5,0,0])]}][{self.colour_mapping[int(self.current_state[5,0,1])]}]")
		print(f"      [{self.colour_mapping[int(self.current_state[5,1,0])]}][{self.colour_mapping[int(self.current_state[5,1,1])]}]")

		print(f"[{self.colour_mapping[int(self.current_state[2,0,0])]}][{self.colour_mapping[int(self.current_state[2,0,1])]}][{self.colour_mapping[int(self.current_state[1,0,0])]}][{self.colour_mapping[int(self.current_state[1,0,1])]}][{self.colour_mapping[int(self.current_state[4,0,0])]}][{self.colour_mapping[int(self.current_state[4,0,1])]}][{self.colour_mapping[int(self.current_state[3,0,0])]}][{self.colour_mapping[int(self.current_state[3,0,1])]}]")
		print(f"[{self.colour_mapping[int(self.current_state[2,1,0])]}][{self.colour_mapping[int(self.current_state[2,1,1])]}][{self.colour_mapping[int(self.current_state[1,1,0])]}][{self.colour_mapping[int(self.current_state[1,1,1])]}][{self.colour_mapping[int(self.current_state[4,1,0])]}][{self.colour_mapping[int(self.current_state[4,1,1])]}][{self.colour_mapping[int(self.current_state[3,1,0])]}][{self.colour_mapping[int(self.current_state[3,1,1])]}]")

		print(f"      [{self.colour_mapping[int(self.current_state[0,0,0])]}][{self.colour_mapping[int(self.current_state[0,0,1])]}]")
		print(f"      [{self.colour_mapping[int(self.current_state[0,1,0])]}][{self.colour_mapping[int(self.current_state[0,1,1])]}]")

	def convert_to_binary(self):
		side_state = np.zeros((6,4,1))
		for side in range(6):
			side_state[side, :, :] = self.current_state[side,:,:].reshape(4,1)
			self.binary_state = OneHotEncoder().fit_transform(side_state.reshape(24,1)).toarray().reshape(-1)

	def F(self):
		temp = np.copy(self.current_state[2,:,1])
		self.current_state[2,:,1] = self.current_state[0,0,:]
		self.current_state[0,0,:] = self.current_state[4,::-1,0]
		self.current_state[4,:,0] = self.current_state[5,1,:]
		self.current_state[5,1,:] = temp[::-1]

		self.current_state[1,:,:] = [[self.current_state[1,1,0],self.current_state[1,0,0]],
									 [self.current_state[1,1,1],self.current_state[1,0,1]]]
		self.history.append("F")

	def Fprime(self):
		temp = np.copy(self.current_state[2,:,1])
		self.current_state[2,:,1] = self.current_state[5,1,::-1]
		self.current_state[5,1,:] = self.current_state[4,:,0]
		self.current_state[4,:,0] = self.current_state[0,0,::-1]
		self.current_state[0,0,:] = temp

		self.current_state[1,:,:] = [[self.current_state[1,0,1],self.current_state[1,1,1]],
									 [self.current_state[1,0,0],self.current_state[1,1,0]]]
		self.history.append("Fprime")	


	def B(self):
		temp = np.copy(self.current_state[5,0,:])
		self.current_state[5,0,:] = self.current_state[4,:,1]
		self.current_state[4,:,1] = self.current_state[0,1,::-1]
		self.current_state[0,1,:] = self.current_state[2,:,0]
		self.current_state[2,:,0] = temp[::-1]

		self.current_state[3,:,:] = [[self.current_state[3,1,0],self.current_state[3,0,0]],
									 [self.current_state[3,1,1],self.current_state[3,0,1]]]
		self.history.append("B")

	def Bprime(self):
		temp = np.copy(self.current_state[5,0,:])
		self.current_state[5,0,:] = self.current_state[2,::-1,0]
		self.current_state[2,:,0] = self.current_state[0,1,:]
		self.current_state[0,1,:] = self.current_state[4,::-1,1]
		self.current_state[4,:,1] = temp

		self.current_state[3,:,:] = [[self.current_state[3,0,1],self.current_state[3,1,1]],
									 [self.current_state[3,0,0],self.current_state[3,1,0]]]
		self.history.append("Bprime")

	def L(self):
		temp = np.copy(self.current_state[3,:,1])
		self.current_state[3,:,1] = self.current_state[0,::-1,0]
		self.current_state[0,:,0] = self.current_state[1,:,0]
		self.current_state[1,:,0] = self.current_state[5,:,0]
		self.current_state[5,:,0] = temp[::-1]

		self.current_state[2,:,:] = [[self.current_state[2,1,0],self.current_state[2,0,0]],
									 [self.current_state[2,1,1],self.current_state[2,0,1]]]
		self.history.append("L")

	def Lprime(self):
		temp = np.copy(self.current_state[3,:,1])
		self.current_state[3,:,1] = self.current_state[5,::-1,0]
		self.current_state[5,:,0] = self.current_state[1,:,0]
		self.current_state[1,:,0] = self.current_state[0,:,0]
		self.current_state[0,:,0] = temp[::-1]

		self.current_state[2,:,:] = [[self.current_state[2,0,1],self.current_state[2,1,1]],
									 [self.current_state[2,0,0],self.current_state[2,1,0]]]
		self.history.append("Lprime")

	def R(self):
		temp = np.copy(self.current_state[5,:,1])
		self.current_state[5,:,1] = self.current_state[1,:,1]
		self.current_state[1,:,1] = self.current_state[0,:,1]
		self.current_state[0,:,1] = self.current_state[3,::-1,0]
		self.current_state[3,:,0] = temp[::-1]

		self.current_state[4,:,:] = [[self.current_state[4,1,0],self.current_state[4,0,0]],
									 [self.current_state[4,1,1],self.current_state[4,0,1]]]
		self.history.append("R")
	
	def Rprime(self):
		temp = np.copy(self.current_state[5,:,1])
		self.current_state[5,:,1] = self.current_state[3,::-1,0]
		self.current_state[3,:,0] = self.current_state[0,::-1,1]
		self.current_state[0,:,1] = self.current_state[1,:,1]
		self.current_state[1,:,1] = temp

		self.current_state[4,:,:] = [[self.current_state[4,0,1],self.current_state[4,1,1]],
									 [self.current_state[4,0,0],self.current_state[4,1,0]]]	
		self.history.append("Rprime")

	def U(self):
		temp = np.copy(self.current_state[2,0,:])
		self.current_state[2,0,:] = self.current_state[1,0,:]
		self.current_state[1,0,:] = self.current_state[4,0,:]
		self.current_state[4,0,:] = self.current_state[3,0,:]
		self.current_state[3,0,:] = temp

		self.current_state[5,:,:] = [[self.current_state[5,1,0],self.current_state[5,0,0]],
									 [self.current_state[5,1,1],self.current_state[5,0,1]]]
		self.history.append("U")

	def Uprime(self):
		temp = np.copy(self.current_state[3,0,:])
		self.current_state[3,0,:] = self.current_state[4,0,:]
		self.current_state[4,0,:] = self.current_state[1,0,:]
		self.current_state[1,0,:] = self.current_state[2,0,:]
		self.current_state[2,0,:] = temp

		self.current_state[5,:,:] = [[self.current_state[5,0,1],self.current_state[5,1,1]],
									 [self.current_state[5,0,0],self.current_state[5,1,0]]]	
		self.history.append("Uprime")

	def D(self):
		temp = np.copy(self.current_state[3,1,:])
		self.current_state[3,1,:] = self.current_state[4,1,:]
		self.current_state[4,1,:] = self.current_state[1,1,:]
		self.current_state[1,1,:] = self.current_state[2,1,:]
		self.current_state[2,1,:] = temp
		
		self.current_state[0,:,:] = [[self.current_state[0,1,0],self.current_state[0,0,0]],
									 [self.current_state[0,1,1],self.current_state[0,0,1]]]
		self.history.append("D")

	def Dprime(self):	
		temp = np.copy(self.current_state[2,1,:])
		self.current_state[2,1,:] = self.current_state[1,1,:]
		self.current_state[1,1,:] = self.current_state[4,1,:]
		self.current_state[4,1,:] = self.current_state[3,1,:]
		self.current_state[3,1,:] = temp	

		self.current_state[0,:,:] = [[self.current_state[0,0,1],self.current_state[0,1,1]],
									 [self.current_state[0,0,0],self.current_state[0,1,0]]]		
		self.history.append("Dprime")
	
	def F2(self):
		self.F()
		self.F()		
		# self.history.append("F2")


	def B2(self):
		self.B()
		self.B()		
		# self.history.append("B2")


	def L2(self):
		self.L()
		self.L()		
		# self.history.append("L2")


	def R2(self):
		self.R()
		self.R()		
		# self.history.append("R2")


	def U2(self):
		self.U()
		self.U()		
		# self.history.append("U2")


	def D2(self):
		self.D()
		self.D()
		# self.history.append("D2")

	def list_execution(self,arr):
		for i in arr:
			self.methods[i]()

	def clear_history(self):
		self.history = []


def generate_sequence(length):
	labels = {"F" : 0, "B" : 1, "L" : 2, "R" : 3, "U" : 4, "D" : 5, "Fprime" :6 , "Bprime" : 7,
		           "Lprime" : 8, "Rprime" : 9, "Uprime" : 10, "Dprime" :11}
	reverse_labels = { 0: "F" ,  1: "B" ,  2: "L" ,  3: "R" ,  4: "U" ,  5: "D" ,  6: "Fprime" ,  7: "Bprime" ,
		            8: "Lprime" ,  9: "Rprime" ,  10: "Uprime" ,  11: "Dprime"}

						# F, B, L, R, U, D, F',B',L',R',U',D'	           
	transition_matrix = [[0 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1],#F
						 [1 ,0 ,1 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1],#B
						 [1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,1 ,1],#L
						 [1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,0 ,0 ,1 ,1],#R
						 [1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,0 ,0],#U
						 [1 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,0 ,0],#D
						 [0 ,0 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1],#Fprime
						 [0 ,0 ,1 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1],#Bprime
						 [1 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1],#Lprime
						 [1 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ,1 ,0 ,1 ,1],#Rprime
						 [1 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ,0 ,1],#Uprime
						 [1 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ,1 ,0]]#Dprime

	transition_matrix = np.array(transition_matrix)
	transformation = []
	transformation_moves =[]
	transformation.append(random.randint(0,11))
	transformation_moves.append(reverse_labels[transformation[0]])

	for i in range(length-1):
		transformation.append(choice(np.where(transition_matrix[transformation[i],:]==1)[0]))
		transformation_moves.append(reverse_labels[transformation[i+1]])

	# print(transformation_moves)
	return transformation_moves, transformation






if __name__ == "__main__":
	cube = rubix2x2()
	cube.print_state()
	cube.convert_to_binary()
	print(cube.binary_state)

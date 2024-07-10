import numpy as np

# Pawn
class Pawn():
	def __init__(self, color, start_file, start_rank):
		"""
		Defining initial attributes of piece

		Starting Position

		File = vertical column (a = 1 = queenside ... h = 8 = kingside)
		Rank = horizontal row (1 = white ... 8 = black)
		self.start_file = start_file
		"""

		# Piece Attributes
		self.name = 'Pawn'		# Name
		self.symbol = ''		# Algebraic notation symbol
		self.value = 1			# Value (1 for pawn)
		self.color = color		# Color
		self.is_active = True	# Active/Inactive

		self.start_rank = start_rank
		# Initializing move counter (increment when moved)
		self.move_count = 0

		# Current Position
		self.file = start_file
		self.rank = start_rank

	def actions(self, piece_list, return_coordinates=False):
		if self.name == 'Pawn':
			action_space = np.zeros((1,56))

			coordinates = []

			if self.is_active:

				if self.color == 'white':
					movement = np.array([[0 ,1],
						  				 [0 ,2],
										 [1 ,1],
										 [-1,1]])

					for i in range(0,4):
							
							# Case (1)
							if i == 0 & 0 < self.file + movement[i,0] < 9 & 0 < self.rank + movement[i,1] < 9:
								blocked = False
								for piece in piece_list:
									if piece.is_active & piece.file == self.file + movement[i,0] & piece.rank == self.rank + movement[i,1]:
										blocked = True
										break
								if blocked == False:
									coordinates.append([self.file + movement[i,0], self.rank + movement[i,1]])
									action_space[0,i] = 1
									
							# Case (2)
							if i == 2 or i == 3:
								if 0 < self.file + movement[i,0] < 9 & 0 < self.rank + movement[i,1] < 9:
									for piece in piece_list:
										if piece.is_active & piece.color != self.color & piece.file == self.file + movement[i,0] & piece.rank == self.rank + movement[i,1]:
											coordinates.append([self.file + movement[i,0], self.rank + movement[i,1]])
											action_space[0,i] = 1
											break
							# Case (3)
							if i == 1 & self.move_count == 0:
								for piece in piece_list:
									blocked = False
									if piece.is_active & piece.file == self.file + movement[i,0] & piece.rank == self.rank + movement[i,1]:
										blocked = True
										break
									elif piece.is_active & piece.file == self.file + movement[i,0] & piece.rank == self.rank + movement[i,1] - 1:
										blocked = True
										break
								if blocked == False:
									coordinates.append([self.file + movement[i,0], self.rank + movement[i,1]])
									action_space[0,i] = 1

				if self.color == 'black':
					movement = np.array([[0 ,-1],
						  				 [0 ,-2],
										 [1 ,-1],
										 [-1,-1]])

					for i in range(0,4):
							if i == 0 & 0 < self.file + movement[i,0] < 9 & 0 < self.rank + movement[i,1] < 9:
								for piece in piece_list:
									blocked = False
									if piece.is_active & piece.file == self.file + movement[i,0] & piece.rank == self.rank + movement[i,1]:
										blocked = True
										break
								if blocked == False:
									coordinates.append([self.file+movement[i,0], self.rank+movement[i,1]])
									action_space[0,i] = 1
							# Case (2)
							if i == 2 or i == 3:
								if 0 < self.file + movement[i,0] < 9 & 0 < self.rank + movement[i,1] < 9:
									for piece in piece_list:
										if piece.is_active & piece.color != self.color & piece.file == self.file + movement[i,0] & piece.rank == self.rank + movement[i,1]:
											coordinates.append([self.file + movement[i,0], self.rank + movement[i,1]])
											action_space[0,i] = 1
											break
							# Case (3)
							if i == 1 & self.move_count == 0:
								for piece in piece_list:
									blocked = False
									if piece.is_active & piece.file == self.file + movement[i,0] & piece.rank == self.rank + movement[i,1]:
										blocked = True
										break
									elif piece.is_active & piece.file == self.file + movement[i,0] & piece.rank == self.rank + movement[i,1] + 1:
										blocked = True
										break
								if blocked == False:
									coordinates.append([self.file + movement[i,0], self.rank + movement[i,1]])
									action_space[0,i] = 1

				# Can pawn promote to queen?
				Promote = False
				# If the pawn is white & has rank 8 or is black & has rank 1, it can promote to queen
				if self.color == 'white' & self.rank == 8:
					Promote = True
				elif self.color == 'black' & self.rank == 1:
					Promote = True
				# If AttackRight is True, append special coordinates
				if Promote:
					coordinates.append([0, 0])
					action_space[0,4] = 1

				# Convert coordinates to numpy array
				coordinates = np.asarray(coordinates)

			# Return possible moves
			if return_coordinates:
				return coordinates
			else:
				return action_space


		# IF THE PIECE IS A QUEEN (HAS BEEN PROMOTED)
		else:
			"""
			The queen's movement is a combination of bishop & rook

			VERTICAL/HORIZONTAL
			For each tile along one of the four movement vectors, append coordinate if:
			(1) The index is in bounds
			(2) There is no piece of the same color
			(3) There was no piece of the opposite color in the preceding step

			Initialize action vector:
			[1-7 +f, 
			 1-7 -f, 
			 1-7 +r, 
			 1-7 -r, 
			 1-7 +f/+r, 
			 1-7 +f/-r, 
			 1-7 -f/+r, 
			 1-7 -f/-r]
			"""
			action_space = np.zeros((1,56))

			# Initialize coordinate aray
			coordinates = []

			if self.is_active:

				# Initialize movement vector array (file, rank)
				movement = np.array([[1 , 0],
						 			 [-1, 0],
									 [0 , 1],
									 [0 ,-1]])

				for i in range(0,4):
					break_loop = False
					for j in range(1,8):
						# Case (1)
						if 0 < self.file + j * movement[i,0] < 9 & 0 < self.rank + j * movement[i,1] < 9:
							for piece in piece_list:
								# Case 2
								if piece.is_active & piece.color == self.color & piece.file == self.file + j * movement[i,0] & piece.rank == self.rank + j * movement[i,1]:
									break_loop = True
								# Case 3
								if piece.is_active & piece.color != self.color & piece.file == self.file + (j-1) * movement[i,0] & piece.rank == self.rank + (j-1) * movement[i,1]:
									break_loop = True
						else: # If the index is no longer in bounds, break
							break
						if break_loop: # If the break_loop was thrown, break
							break
						# If the break_loop was not thrown, append coordinates
						coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
						action_space[0,7*i+(j-1)] = 1

				# DIAGONAL
				# For each tile along one of the four movement vectors, append coordinate if:
				# (1) The index is in bounds
				# (2) There is no piece of the same color
				# (3) There was no piece of the opposite color in the preceding step

				# Initialize movement vector array (file, rank)
				movement = np.array([[1 , 1],
						 			 [1 ,-1],
									 [-1, 1],
									 [-1,-1]])

				for i in range(0,4):
					break_loop = False
					for j in range(1,8):
						# Case (1)
						if 0 < self.file + j * movement[i,0] < 9 & 0 < self.rank + j * movement[i,1] < 9:
							for piece in piece_list:
								# Case 2
								if piece.is_active & piece.color == self.color & piece.file == self.file + j * movement[i,0] & piece.rank == self.rank + j * movement[i,1]:
									break_loop = True
								# Case 3
								if piece.is_active & piece.color != self.color & piece.file == self.file + (j-1) * movement[i,0] & piece.rank == self.rank + (j-1) * movement[i,1]:
									break_loop = True
						else: # If the index is no longer in bounds, break
							break
						if break_loop: # If the break_loop was thrown, break
							break
						# If the break_loop was not thrown, append coordinates
						coordinates.append([self.file + j * movement[i,0], self.rank + j * movement[i,1]])
						action_space[0,7 * i + (j-1) + 28] = 1

				# Convert coordinates to numpy array
				coordinates = np.asarray(coordinates)

			# Return possible moves
			if return_coordinates:
				return coordinates
			else:
				return action_space


	def move(self, action, piece_list, print_move=False, algebraic=True):

		"""Moving piece's position

		Args:		action (element of action vector), 
					piece list, 
					print move (bool),
					algebraic notation (bool)
		"""
		# Initializing placeholder
		promoted = False
		# Temporarily save old position for the purposes of algebraic notation
		old_rank = self.rank
		old_file = self.file

		# IF THE PIECE IS A PAWN (HAS NOT BEEN PROMOTED)
		if self.name == "Pawn":

			# Action vector:
			# [1 forward, 2 forward, attack (+file), attack (-file), promotion, 51 zeros]

			# Move 1 forward
			if action == 0:
				if self.color == 'white':
					self.rank = self.rank+1
				else:
					self.rank = self.rank-1
			# Move 2 forward
			elif action == 1:
				if self.color == 'white':
					self.rank = self.rank+2
				else:
					self.rank = self.rank-2
			# Attack (+file)
			elif action == 2:
				if self.color == 'white':
					self.file = self.file+1
					self.rank = self.rank+1
				else:
					self.file = self.file+1
					self.rank = self.rank-1
			# Attack (-file)
			elif action == 3:
				if self.color == 'white':
					self.file = self.file-1
					self.rank = self.rank+1
				else:
					self.file = self.file-1
					self.rank = self.rank-1
			# Promote to queen
			else:
				promoted = True
				self.name = 'Queen'
				self.symbol = 'Q'
				self.value = 9


		# IF THE PIECE IS A QUEEN (HAS BEEN PROMOTED)
		else:
			"""
			Action vector:
			[1-7 +f, 
			 1-7 -f, 
			 1-7 +r, 
			 1-7 -r, 
			 1-7 +f/+r, 
			 1-7 +f/-r, 
			 1-7 -f/+r, 
			 1-7 -f/-r]
			"""

			# +file movements
			if 0 <= action < 7:
				self.file = self.file + (action+1)
			# -file movements
			elif 7 <= action < 14:
				self.file = self.file - (action-6)
			# +rank movements
			elif 14 <= action < 21:
				self.rank = self.rank + (action-13)
			# -rank movements
			elif 21 <= action < 28:
				self.rank = self.rank - (action-20)
			# +f/+r movements
			elif 28 <= action < 35:
				self.file = self.file + (action-27)
				self.rank = self.rank + (action-27)
			# +f/-r movements
			elif 35 <= action < 42:
				self.file = self.file + (action-34)
				self.rank = self.rank - (action-34)
			# -f/+r movements
			elif 42 <= action < 49:
				self.file = self.file - (action-41)
				self.rank = self.rank + (action-41)
			# -f/-r movements
			else:
				self.file = self.file - (action-48)
				self.rank = self.rank - (action-48)


		# Increment move counter
		self.move_count += 1

		# If a piece was in the destination tile, remove the piece
		piece_remove = False
		for piece in piece_list:
			if piece.is_active & piece.color != self.color & piece.file == self.file & piece.rank == self.rank:
				piece.remove()
				piece_remove = True
				remove_name = piece.name
				break

		# Print movement if indicated
		file_list = ['a','b','c','d','e','f','g','h']
		if print_move:
			if algebraic:
				if promoted:
					print(f"\n{file_list[self.file-1]}{self.rank}={self.symbol}", end=" "*20+"\r")
				elif piece_remove:
					print(f"\n{self.symbol}{file_list[old_file-1]}{old_rank} x {file_list[self.file-1]}{self.rank}", end=" "*20+"\r")
				else:
					print(f"\n{self.symbol}{file_list[old_file-1]}{old_rank}-{file_list[self.file-1]}{self.rank}", end=" "*20+"\r")
			else:
				if promoted:
					print(f"\n{self.name} promoted to Queen", end=" "*20+"\r")
				elif piece_remove:
					print(f"\n{self.name} to {self.file},{self.rank} taking {remove_name}", end=" "*20+"\r")
				else:
					print(f"\n{self.name} to {self.file},{self.rank}", end=" "*20+"\r")


	def remove(self):

		"""Removing piece from board"""

		# Args:	none
		# Returns:	void
		self.is_active = False

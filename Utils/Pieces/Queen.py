import numpy as np

class Queen():

	"""Defining attributes of queen piece"""

	def __init__(self, color):

		"""Defining initial attributes of piece"""

		# Piece Attributes
		self.name = 'Queen'		# Name
		self.symbol = 'Q'		# Symbol for algebraic notation
		self.value = 9			# Value (9 for queen)
		self.color = color		# Color
		self.is_active = True	# Active/Inactive

		# Starting position
		# File = vertical column (a = 1 = queenside ... h = 8 = kingside)
		# Rank = horizontal row (1 = white ... 8 = black)
		if color == 'white':
			self.start_file = 4
			self.start_rank = 1
		else:
			self.start_file = 4
			self.start_rank = 8
		# Initializing move counter (increment when moved)
		self.move_count = 0

		# Current position
		self.file = self.start_file
		self.rank = self.start_rank


	# Returning numpy array with possible actions for piece
	# Array format:
	# [[file1 rank1]
	#  [file2 rank2]...]
	def actions(self, piece_list, return_coordinates=False):

		"""Determining possible actions for piece"""

		# Requires: piece_list
		# Returns: numpy array

		# The queen's movement is a combination of bishop and rook

		# VERTICAL/HORIZONTAL
		# For each tile along one of the four movement vectors, append coordinate if:
		# (1) The index is in bounds
		# (2) There is no piece of the same color
		# (3) There was no piece of the opposite color in the preceding step

		# Initialize action vector:
		# [1-7 +f, 1-7 -f, 1-7 +r, 1-7 -r, 1-7 +f/+r, 1-7 +f/-r, 1-7 -f/+r, 1-7 -f/-r]
		action_space = np.zeros((1,56))

		# Initialize coordinate aray
		coordinates = []

		if self.is_active:

			# Initialize movement vector array (file, rank)
			movement = np.array([[1,0],[-1,0],[0,1],[0,-1]])

			for i in range(0,4):
				break_loop = False
				for j in range(1,8):
					# Condition (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Condition 2
							if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
								break_loop = True
							# Condition 3
							if piece.is_active and piece.color != self.color and piece.file == self.file+(j-1)*movement[i,0] and piece.rank == self.rank+(j-1)*movement[i,1]:
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
			movement = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])

			for i in range(0,4):
				break_loop = False
				for j in range(1,8):
					# Condition (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Condition 2
							if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
								break_loop = True
							# Condition 3
							if piece.is_active and piece.color != self.color and piece.file == self.file+(j-1)*movement[i,0] and piece.rank == self.rank+(j-1)*movement[i,1]:
								break_loop = True
					else: # If the index is no longer in bounds, break
						break
					if break_loop: # If the break_loop was thrown, break
						break
					# If the break_loop was not thrown, append coordinates
					coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
					action_space[0,7*i+(j-1)+28] = 1

			# Convert coordinates to numpy array
			coordinates = np.asarray(coordinates)

		# Return possible moves
		if return_coordinates:
			return coordinates
		else:
			return action_space


	def move(self, action, piece_list, print_move=False, algebraic=True):

		"""Moving piece's position"""

		# Requires:	(1) action (element of action vector), (2) piece list, (3) print move? (4) algebraic notation?
		# Returns:	void

		# Action vector:
		# [1-7 +f, 1-7 -f, 1-7 +r, 1-7 -r, 1-7 +f/+r, 1-7 +f/-r, 1-7 -f/+r, 1-7 -f/-r]

		# Temporarily save old position for the purposes of algebraic notation
		old_rank = self.rank
		old_file = self.file

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


		# Update move counter
		self.move_count += 1

		# If a piece was in the destination tile, remove the piece
		piece_remove = False
		for piece in piece_list:
			if piece.is_active and piece.color != self.color and piece.file == self.file and piece.rank == self.rank:
				piece.remove()
				piece_remove = True
				remove_name = piece.name
				break

		# Print movement if indicated
		file_list = ['a','b','c','d','e','f','g','h']
		if print_move:
			if algebraic:
				if piece_remove:
					print(f"{self.symbol}{file_list[old_file-1]}{old_rank} x {file_list[self.file-1]}{self.rank}", end=" "*20+"\r\n")
				else:
					print(f"{self.symbol}{file_list[old_file-1]}{old_rank}-{file_list[self.file-1]}{self.rank}", end=" "*20+"\r\n")
			else:
				if piece_remove:
					print(f"{self.name} to {self.file},{self.rank} taking {remove_name}", end=" "*20+"\r\n")
				else:
					print(f"{self.name} to {self.file},{self.rank}", end=" "*20+"\r\n")


	def remove(self):

		"""Removing piece from board"""

		# Requires:	none
		# Returns:	void
		self.is_active = False
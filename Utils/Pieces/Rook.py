import numpy as np

class Rook():

	"""Defining attributes of rook piece"""

	def __init__(self, color, start_file, start_rank):

		"""
		Initialize a Rook piece with its initial attributes.

		Parameters:
		color (str): The color of the piece ('white' or 'black').
		start_file (int): The initial vertical column of the piece (a = 1 = queenside, h = 8 = kingside).
		start_rank (int): The initial horizontal row of the piece (1 = white, 8 = black).

		Returns:
		None
		"""

		# Piece Attributes
		self.name = 'Rook'		# Name
		self.symbol = 'R'		# Sybmol for algebraic notation
		self.value = 5			# Value (5 for rook)
		self.color = color		# Color
		self.is_active = True	# Active/Inactive

		# Starting position
		# File = vertical column (a = 1 = queenside ... h = 8 = kingside)
		# Rank = horizontal row (1 = white ... 8 = black)
		self.start_file = start_file
		self.start_rank = start_rank
		# Initializing move counter (increment when moved)
		self.move_count = 0

		# Current position
		self.file = start_file
		self.rank = start_rank


	# Returning numpy array with possible actions for piece
	# Array format:
	# [[file1 rank1]
	#  [file2 rank2]...]
	def actions(self, piece_list, return_coordinates=False):

		"""
		Determines the possible actions for the rook piece.

		Parameters:
		piece_list (list): A list of all pieces on the board.
		return_coordinates (bool, optional): If True, returns the coordinates of possible moves. Defaults to False.

		Returns:
		numpy.ndarray: A 1x56 numpy array representing the possible actions for the rook. If return_coordinates is True,
		returns a 2D numpy array with the coordinates of possible moves.
		"""

		# The rook may move any number of spaces along its current rank/file.
		# It may also attack opposing pieces in its movement path.

		# For each tile along one of the four movement vectors, append coordinate if:
		# (1) The index is in bounds
		# (2) There is no piece of the same color
		# (3) There was no piece of the opposite color in the preceding step

		# Initialize action vector:
		# [1-7 +file, 1-7 -file, 1-7 +rank, 1-7 -rank, 28 zeros]
		action_space = np.zeros((1,56))

		# Initialize coordinate aray
		coordinates = []

		if self.is_active:

			# Initialize movement vector array (file, rank)
			movement = np.array([[1, 0], [-1, 0], 
								 [0, 1], [0 ,-1]])

			for i in range(0,4):
				break_loop = False
				for j in range(1,8):
					# Case (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Case 2
							if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
								break_loop = True
							# Case 3
							if piece.is_active and piece.color != self.color and piece.file == self.file+(j-1)*movement[i,0] and piece.rank == self.rank+(j-1)*movement[i,1]:
								break_loop = True
					else: # If the index is no longer in bounds, break
						break
					if break_loop: # If the break_loop was thrown, break
						break
					# If the break_loop was not thrown, append coordinates
					coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
					action_space[0,7*i+(j-1)] = 1

			# Convert coordinates to numpy array
			coordinates = np.asarray(coordinates)

		# Return possible moves
		if return_coordinates:
			return coordinates
		else:
			return action_space


	def move(self, action, piece_list, print_move=False, algebraic=True):

		"""
		Moves the piece's position based on the given action and piece list.

		Parameters:
		action (int): An element of the action vector representing the movement direction.
		piece_list (list): A list of all pieces on the board.
		print_move (bool, optional): If True, prints the move in a human-readable format. Defaults to False.
		algebraic (bool, optional): If True, prints the move in algebraic notation. Defaults to True.

		Returns:
		None
		"""

		# Action vector:
		# [1-7 +file, 1-7 -file, 1-7 +rank, 1-7 -rank, 28 zeros]

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
		else:
			self.rank = self.rank - (action-20)


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
					print(f"\n{self.symbol}{file_list[old_file-1]}{old_rank} x {file_list[self.file-1]}{self.rank}", end=" "*20+"\r")
				else:
					print(f"\n{self.symbol}{file_list[old_file-1]}{old_rank}-{file_list[self.file-1]}{self.rank}", end=" "*20+"\r") #
			else:
				if piece_remove:
					print(f"\n{self.name} to {self.file},{self.rank} taking {remove_name}", end=" "*20+"\r")
				else:
					print(f"\n{self.name} to {self.file},{self.rank}", end=" "*20+"\r")


	def remove(self):
		"""
		Removes the piece from the board.

		This method sets the 'is_active' attribute of the piece to False, effectively
		marking it as inactive and removing it from the game.

		Args:
			None

		Returns:
			None
		"""
		self.is_active = False
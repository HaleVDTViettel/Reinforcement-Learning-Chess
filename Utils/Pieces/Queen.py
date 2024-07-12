import numpy as np

class Queen():

	"""Defining attributes of queen piece"""

	def __init__(self, color):

		"""
		Initialize a new instance of the Queen class.

		Parameters:
		color (str): The color of the queen piece. It should be either 'white' or 'black'.

		Attributes:
		name (str): The name of the piece, which is 'Queen'.
		symbol (str): The symbol for algebraic notation, which is 'Q'.
		value (int): The value of the piece, which is 9 for a queen.
		color (str): The color of the piece.
		is_active (bool): Indicates whether the piece is active or not.
		start_file (int): The starting file position of the piece.
		start_rank (int): The starting rank position of the piece.
		move_count (int): The number of times the piece has been moved.
		file (int): The current file position of the piece.
		rank (int): The current rank position of the piece.
		"""

		# Piece Attributes
		self.name = 'Queen'
		self.symbol = 'Q'
		self.value = 9
		self.color = color
		self.is_active = True

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

	def actions(self, piece_list, return_coordinates=False):

		"""
		Determines the possible actions for the queen piece.

		Args:
		piece_list (list): A list of all pieces on the board.
		return_coordinates (bool, optional): If True, returns the coordinates of possible moves. Defaults to False.

		Returns:
		numpy.ndarray or list: A numpy array of action space if return_coordinates is False, else a list of coordinates.

		The queen's movement is a combination of bishop and rook. This function calculates the possible moves for the queen
		based on the rules of chess. It checks for valid moves in all eight directions (forward, backward, left, right,
		and diagonally) and returns either the action space (numpy array) or the coordinates of possible moves.
		"""
		# Initialize action vector:
		# [1-7 +f, 1-7 -f, 1-7 +r, 1-7 -r, 1-7 +f/+r, 1-7 +f/-r, 1-7 -f/+r, 1-7 -f/-r]
		action_space = np.zeros((1,56))

		# Initialize coordinate aray
		coordinates = []

		if self.is_active:

			# Initialize movement vector array (file, rank)
			movement = np.array([[1,0],
								 [-1,0],
								 [0,1],
								 [0,-1]])

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
					action_space[0,7*i+(j-1)+28] = 1

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
					print(f"\n{self.symbol}{file_list[old_file-1]}{old_rank} x {file_list[self.file-1]}{self.rank}", end=" "*20+"\r")
				else:
					print(f"\n{self.symbol}{file_list[old_file-1]}{old_rank}-{file_list[self.file-1]}{self.rank}", end=" "*20+"\r")
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
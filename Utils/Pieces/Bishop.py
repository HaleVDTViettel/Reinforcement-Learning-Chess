import numpy as np

class Bishop():

	"""Defining attributes of bishop piece"""

	def __init__(self, color, start_file, start_rank):

		"""
		Initialize a Bishop piece with its initial attributes.

		Parameters:
		color (str): The color of the bishop piece ('white' or 'black').
		start_file (int): The initial vertical column of the bishop piece (1-8, where 1 is queenside and 8 is kingside).
		start_rank (int): The initial horizontal row of the bishop piece (1-8, where 1 is white and 8 is black).

		Returns:
		None
		"""

		# Piece Attributes
		self.name = 'Bishop'	# Name
		self.symbol = 'B'		# Symbol for algebraic notation
		self.value = 3			# Value (3 for bishop)
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


	
	def actions(self, piece_list, return_coordinates=False):
		"""
		Calculate the possible actions for the bishop piece.

		Args:
		piece_list (list): A list of all pieces on the board.
		return_coordinates (bool, optional): If True, return the coordinates of the possible actions. Defaults to False.

		Returns:
		numpy.ndarray: A 1x56 numpy array representing the possible actions for the bishop. If return_coordinates is True,
		return a 2D numpy array with the coordinates of the possible actions.

		The bishop can move diagonally in any direction. For each tile along one of the four movement vectors, append coordinate if:
		(1) The index is in bounds
		(2) There is no piece of the same color
		(3) There was no piece of the opposite color in the preceding step

		Initialize action vector:
		[1-7 +f/+r, 1-7 +f/-r, 1-7 -f/+r, 1-7 -f/-r, 28 zeros]
		"""
		action_space = np.zeros((1,56))

		# Initialize coordinate aray
		coordinates = []

		if self.is_active:

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
		This function moves the piece's position based on the given action and piece list.
		It also handles the removal of a piece if one is present in the destination tile.
		If print_move is True, it prints the movement in either algebraic or descriptive notation.

		Args:
			action (int): An element of the action vector representing the movement direction.
			piece_list (list): A list of all pieces on the board.
			print_move (bool, optional): A flag indicating whether to print the movement. Defaults to False.
			algebraic (bool, optional): A flag indicating whether to use algebraic notation for printing. Defaults to True.

		Action vector:
		[1-7 +f/+r, 
		1-7 +f/-r, 
		1-7 -f/+r, 
		1-7 -f/-r, 
		28 zeros]

		Returns:
			None
		"""
		old_rank = self.rank
		old_file = self.file

		# +f/+r movements
		if 0 <= action < 7:
			self.file = self.file + (action+1)
			self.rank = self.rank + (action+1)
		elif 7 <= action < 14:
			self.file = self.file + (action-6)
			self.rank = self.rank - (action-6)
		elif 14 <= action < 21:
			self.file = self.file - (action-13)
			self.rank = self.rank + (action-13)
		else:
			self.file = self.file - (action-20)
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
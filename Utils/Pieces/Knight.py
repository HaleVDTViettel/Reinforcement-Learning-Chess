import numpy as np

class Knight():

	"""Defining attributes of knight piece"""

	def __init__(self, color, start_file, start_rank):

		"""
		Initialize a Knight piece with its initial attributes.

		Parameters:
		color (str): The color of the knight piece ('white' or 'black').
		start_file (int): The initial vertical column of the knight piece (1-8, where 'a'=1 and 'h'=8).
		start_rank (int): The initial horizontal row of the knight piece (1-8, where 1 is white and 8 is black).

		Returns:
		None
		"""

		# Piece Attributes
		self.name = 'Knight'	# Name
		self.symbol = 'N'		# Symbol for algebraic notation
		self.value = 3			# Value (3 for knight)
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
		Determines the possible actions for the knight piece.

		Parameters:
		piece_list (list): A list of all pieces on the board.
		return_coordinates (bool, optional): If True, return the coordinates of possible moves.
											If False, return a binary action space vector.
											Defaults to False.

		Returns:
		numpy.ndarray: A numpy array containing either the coordinates of possible moves or a binary action space vector.
		"""

		# A knight may have any of 8 possible actions:
		# Move forward 2 tiles in any direction + 1 tile perpendicularly

		# For each of the 8 possible actions, if:
		# (1) The index is not out of bounds
		# (2) There is not a piece of the same color
		# Then append the coordinates to the output array

		# Initialize action vector:
		# [8 knight moves, 48 zeros]
		action_space = np.zeros((1,56))

		# Initialize coordinate aray
		coordinates = []

		if self.is_active:

			# Initialize movement array (file, rank)
			movement = np.array([[2,1],[2,-1],[1,2],[-1,2],[1,-2],[-1,-2],[-2,1],[-2,-1]])
			for i in range(0,8):
				continue_loop = False
				# Case (1)
				if 0 < self.file+movement[i,0] < 9 and 0 < self.rank+movement[i,1] < 9:
					for piece in piece_list:
						# Case (2)
						if piece.is_active and piece.color == self.color and piece.file == self.file+movement[i,0] and piece.rank == self.rank+movement[i,1]:
							continue_loop = True
							break
				else: # If the index is not in bounds, continue
					continue
				# If continue_loop is True, continue the loop without appending coordinate. Else, append coordinate.
				if continue_loop:
					continue
				coordinates.append([self.file+movement[i,0], self.rank+movement[i,1]])
				action_space[0,i] = 1


			# Convert coordinates to numpy array
			coordinates = np.asarray(coordinates)

		# Return possible moves
		if return_coordinates:
			return coordinates
		else:
			return action_space


	def move(self, action, piece_list, print_move=False, algebraic=True):
		"""
		Moves the piece's position based on the given action.

		Parameters:
		action (int): An element of the action vector representing the desired move.
		piece_list (list): A list of all pieces on the board.
		print_move (bool, optional): If True, prints the move in a human-readable format. Defaults to False.
		algebraic (bool, optional): If True, prints the move in algebraic notation. If False, prints the move in a descriptive format. Defaults to True.

		Returns:
		None
		"""

		# Temporarily save old position for the purposes of algebraic notation
		old_rank = self.rank
		old_file = self.file

		# Action vector:
		# [[2,1],[2,-1],[1,2],[-1,2],[1,-2],[-1,-2],[-2,1],[-2,-1]]

		if action == 0:
			self.file = self.file + 2
			self.rank = self.rank + 1
		elif action == 1:
			self.file = self.file + 2
			self.rank = self.rank - 1
		elif action == 2:
			self.file = self.file + 1
			self.rank = self.rank + 2
		elif action == 3:
			self.file = self.file - 1
			self.rank = self.rank + 2
		elif action == 4:
			self.file = self.file + 1
			self.rank = self.rank - 2
		elif action == 5:
			self.file = self.file - 1
			self.rank = self.rank - 2
		elif action == 6:
			self.file = self.file - 2
			self.rank = self.rank + 1
		else:
			self.file = self.file - 2
			self.rank = self.rank - 1

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
import numpy as np

class Knight():

	"""Defining attributes of knight piece"""

	def __init__(self, color, start_file, start_rank):

		"""Defining initial attributes of piece"""

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

		"""Determining possible actions for piece"""

		# Requires: piece_list
		# Returns: numpy array

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
				# Condition (1)
				if 0 < self.file+movement[i,0] < 9 and 0 < self.rank+movement[i,1] < 9:
					for piece in piece_list:
						# Condition (2)
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

		"""Moving piece's position"""

		# Requires:	(1) action (element of action vector), (2) piece list, (3) print move? (4) algebraic notation?
		# Returns:	void

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
		if print_move and algebraic:
			if piece_remove:
				print(self.symbol + file_list[old_file-1] + str(old_rank)+ " x " + file_list[self.file-1] + str(self.rank))
			else:
				print(self.symbol + file_list[old_file-1] + str(old_rank) + "-" + file_list[self.file-1] + str(self.rank))
		elif print_move:
			if piece_remove:
				print(self.name + " to " + str(self.file) + "," + str(self.rank) + " taking " + remove_name)
			else:
				print(self.name + " to " + str(self.file) + "," + str(self.rank))


	def remove(self):

		"""Removing piece from board"""

		# Requires:	none
		# Returns:	void
		self.is_active = False
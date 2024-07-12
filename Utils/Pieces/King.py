import numpy as np

class King():

	"""Defining attributes of king piece"""

	def __init__(self, color):

		"""
		Initialize the King piece with its initial attributes.

		Parameters:
		color (str): The color of the piece ('white' or 'black').

		Attributes:
		name (str): The name of the piece, always 'King'.
		symbol (str): The symbol for algebraic notation, always 'K'.
		value (int): The value of the piece, always 100.
		color (str): The color of the piece ('white' or 'black').
		is_active (bool): Indicates whether the piece is active or not, always True.
		start_file (int): The starting file of the piece (1-8).
		start_rank (int): The starting rank of the piece (1-8).
		move_count (int): The number of times the piece has been moved.
		file (int): The current file of the piece (1-8).
		rank (int): The current rank of the piece (1-8).
		kCastle (bool): Indicates whether the king can perform kingside castle.
		qCastle (bool): Indicates whether the king can perform queenside castle.
		"""

		# Piece Attributes
		self.name = 'King'		# Name
		self.symbol = 'K'		# Symbol for algebraic notation
		self.value = 100		# Value
		self.color = color		# Color
		self.is_active = True	# Active/Inactive

		# Starting position
		# File = vertical column (a = 1 = queenside ... h = 8 = kingside)
		# Rank = horizontal row (1 = white ... 8 = black)
		if color == 'white':
			self.start_file = 5
			self.start_rank = 1
		else:
			self.start_file = 5
			self.start_rank = 8
		self.move_count = 0

		# Current position
		self.file = self.start_file
		self.rank = self.start_rank

		# Special attributes
		# Can kingside castle?
		self.kCastle = False
		# Can queenside castle?
		self.qCastle = False

	def actions(self, piece_list, return_coordinates=False):

		"""
		Determines possible actions for a piece.

		Args:
		piece_list (list): A list of all pieces on the board.
		return_coordinates (bool, optional): A boolean indicating whether to return the coordinates of possible moves or the action space. Default is False.

		Returns:
		numpy.ndarray: A numpy array representing the possible moves or the action space.
		"""

		# The king may move one tile in any direction. The king may castle as a first move.
		# Special case of the queen where "j" is fixed to 1.

		# VERTICAL/HORIZONTAL
		# For each tile along one of the four movement vectors, append coordinate if:
		# (1) The index is in bounds
		# (2) There is no piece of the same color

		# Initialize action vector:
		# [8 king moves, kingside castle, queenside castle, 46 zeros]
		action_space = np.zeros((1,56))

		# Initialize coordinate aray
		coordinates = []

		if self.is_active:

			# Initialize movement vector array (file, rank)
			movement = np.array([[1,0],[-1,0],[0,1],[0,-1]])

			for i in range(0,4):
				break_loop = False
				for j in range(1,2):
					# Case (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Case 2
							if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
								break_loop = True
					else: # If the index is no longer in bounds, break
						break
					if break_loop: # If the break_loop was thrown, break
						break
					# If the break_loop was not thrown, append coordinates
					coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
					action_space[0,i] = 1

			# DIAGONAL
			# For each tile along one of the four movement vectors, append coordinate if:
			# (1) The index is in bounds
			# (2) There is no piece of the same color

			# Initialize movement vector array (file, rank)
			movement = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])

			for i in range(0,4):
				break_loop = False
				for j in range(1,2):
					# Case (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Case 2
							if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
								break_loop = True
					else: # If the index is no longer in bounds, break
						break
					if break_loop: # If the break_loop was thrown, break
						break
					# If the break_loop was not thrown, append coordinates
					coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
					action_space[0,i+4] = 1


			# Can king perform kingside castle?
			Castle = True

			# Conditions:
			# (1) If the king and kingside rook have not moved
			# (2) There are no pieces in between them
			# (3) The king is not in check

			# Case (1)
			if self.move_count == 0 and ((self.color == 'white' and piece_list[7].move_count == 0) or (self.color == 'black' and piece_list[31].move_count == 0)):
				for piece in piece_list:
					# Case (2)
					if piece.is_active and piece.rank == self.rank and (piece.file == self.file+1 or piece.file == self.file+2):
						Castle = False
						break
					# Case (3)
					elif piece.is_active and piece.name != 'King' and piece.color != self.color and piece.actions(piece_list, True).size > 0 and (piece.actions(piece_list, True) == np.array([self.file, self.rank])).all(1).any():
						Castle = False
						break
					else:
						Castle = True
			else:
				Castle = False

			if Castle:
				self.kCastle = True
				coordinates.append([0, 0])
				action_space[0,8] = 1
			else:
				self.kCastle = False

			# Can king perform queenside castle?
			Castle = True

			# Conditions:
			# (1) If the king and queenside rook have not moved
			# (2) There are no pieces in between them
			# (3) The king is not in check

			# Case (1)
			if self.move_count == 0 and ((self.color == 'white' and piece_list[0].move_count == 0) or (self.color == 'black' and piece_list[24].move_count == 0)):
				for piece in piece_list:
					# Case (2)
					if piece.is_active and piece.rank == self.rank and (piece.file == self.file-1 or piece.file == self.file-2 or piece.file == self.file-3):
						Castle = False
						break
					# Case (3)
					elif piece.is_active and piece.name != 'King' and piece.color != self.color and piece.actions(piece_list, True).size > 0 and (piece.actions(piece_list, True) == np.array([self.file, self.rank])).all(1).any():
						Castle = False
						break
					else:
						Castle = True
			else:
				Castle = False

			if Castle:
				self.qCastle = True
				coordinates.append([-1,-1])
				action_space[0,9] = 1
			else:
				self.qCastle = False

			# Convert coordinates to numpy array
			coordinates = np.asarray(coordinates)

		# Return possible moves
		if return_coordinates:
			return coordinates
		else:
			return action_space


	def move(self, action, piece_list, print_move=False, algebraic=True):

		"""
		Moves the piece's position based on the given action and the piece list.

		Parameters:
		action (int): The action to be taken by the piece. This is an element of the action vector.
		piece_list (list): A list of all pieces on the board.
		print_move (bool, optional): A boolean indicating whether to print the move. Default is False.
		algebraic (bool, optional): A boolean indicating whether to use algebraic notation for printing the move. Default is True.

		Returns:
		None
		"""

		# Temporarily save old position for the purposes of algebraic notation
		old_rank = self.rank
		old_file = self.file
		# Initializing placeholders
		kcastle = False
		qcastle = False

		# Action vector:
		# [8 king moves, kingside castle, queenside castle, 46 zeros]
		# [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1], kC, qC]

		if action == 0:
			self.file = self.file + 1
		elif action == 1:
			self.file = self.file - 1
		elif action == 2:
			self.rank = self.rank + 1
		elif action == 3:
			self.rank = self.rank - 1
		elif action == 4:
			self.file = self.file + 1
			self.rank = self.rank + 1
		elif action == 5:
			self.file = self.file + 1
			self.rank = self.rank - 1
		elif action == 6:
			self.file = self.file - 1
			self.rank = self.rank + 1
		elif action == 7:
			self.file = self.file - 1
			self.rank = self.rank - 1
		# Kingside castle
		elif action == 8:
			kcastle = True
			self.file = self.file + 2
			if self.color == 'white':
				piece_list[7].file = piece_list[7].file - 2
			else:
				piece_list[31].file = piece_list[31].file - 2
		# Queenside castle
		else:
			qcastle = True
			self.file = self.file - 2
			if self.color == 'white':
				piece_list[0].file = piece_list[0].file + 3
			else:
				piece_list[24].file = piece_list[24].file + 3

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
				if kcastle:
					print("0-0", end=" "*20+"\r")
				elif qcastle:
					print("0-0-0", end=" "*20+"\r")
				elif piece_remove:
					print(f"\n{self.symbol}{file_list[old_file-1]}{old_rank} x {file_list[self.file-1]}{self.rank}", end=" "*20+"\r")
				else:
					print(f"\n{self.symbol}{file_list[old_file-1]}{old_rank}-{file_list[self.file-1]}{self.rank}", end=" "*20+"\r")
			else:
				if kcastle:
					print("\nKingside Castle", end=" "*20+"\r")
				elif qcastle:
					print("\nQueenside Castle", end=" "*20+"\r")
				elif piece_remove:
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
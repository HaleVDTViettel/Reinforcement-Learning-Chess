import numpy as np

class King():

	"""Defining attributes of king piece"""

	def __init__(self, color):

		"""Defining initial attributes of piece"""

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
		# Initializing move counter (increment when moved)
		self.move_count = 0

		# Current position
		self.file = self.start_file
		self.rank = self.start_rank

		# Special attributes
		# Can kingside castle?
		self.kCastle = False
		# Can queenside castle?
		self.qCastle = False


	# Returning numpy array with possible actions for piece
	# Array format:
	# [[file1 rank1]
	#  [file2 rank2]...]
	def actions(self, piece_list, return_coordinates=False):

		"""Determining possible actions for piece"""

		# Requires: piece_list
		# Returns: numpy array

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
					# Condition (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Condition 2
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
					# Condition (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Condition 2
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

			# Condition (1)
			if self.move_count == 0 and ((self.color == 'white' and piece_list[7].move_count == 0) or (self.color == 'black' and piece_list[31].move_count == 0)):
				for piece in piece_list:
					# Condition (2)
					if piece.is_active and piece.rank == self.rank and (piece.file == self.file+1 or piece.file == self.file+2):
						Castle = False
						break
					# Condition (3)
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

			# Condition (1)
			if self.move_count == 0 and ((self.color == 'white' and piece_list[0].move_count == 0) or (self.color == 'black' and piece_list[24].move_count == 0)):
				for piece in piece_list:
					# Condition (2)
					if piece.is_active and piece.rank == self.rank and (piece.file == self.file-1 or piece.file == self.file-2 or piece.file == self.file-3):
						Castle = False
						break
					# Condition (3)
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

		"""Moving piece's position"""

		# Requires:	(1) action (element of action vector), (2) piece list, (3) print move? (4) algebraic notation?
		# Returns:	void

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
		if print_move and algebraic:
			if kcastle:
				print("0-0")
			elif qcastle:
				print("0-0-0")
			elif piece_remove:
				print(self.symbol + file_list[old_file-1] + str(old_rank)+ " x " + file_list[self.file-1] + str(self.rank))
			else:
				print(self.symbol + file_list[old_file-1] + str(old_rank) + "-" + file_list[self.file-1] + str(self.rank))
		elif print_move:
			if kcastle:
				print("Kingside Castle")
			elif qcastle:
				print("Queenside Castle")
			elif piece_remove:
				print(self.name + " to " + str(self.file) + "," + str(self.rank) + " taking " + remove_name)
			else:
				print(self.name + " to " + str(self.file) + "," + str(self.rank))

		if print_move:
			if algebraic:
				if kcastle:
					print("0-0", end=" "*20+"\r\n")
				elif qcastle:
					print("0-0-0", end=" "*20+"\r\n")
				elif piece_remove:
					print(f"{self.symbol}{file_list[old_file-1]}{old_rank} x {file_list[self.file-1]}{self.rank}", end=" "*20+"\r\n")
				else:
					print(f"{self.symbol}{file_list[old_file-1]}{old_rank}-{file_list[self.file-1]}{self.rank}", end=" "*20+"\r\n")
			else:
				if kcastle:
					print("Kingside Castle", end=" "*20+"\r\n")
				elif qcastle:
					print("Queenside Castle", end=" "*20+"\r\n")
				elif piece_remove:
					print(f"{self.name} to {self.file},{self.rank} taking {remove_name}", end=" "*20+"\r\n")
				else:
					print(f"{self.name} to {self.file},{self.rank}", end=" "*20+"\r\n")



	def remove(self):

		"""Removing piece from board"""

		# Requires:	none
		# Returns:	void
		self.is_active = False
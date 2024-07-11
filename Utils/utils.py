import numpy as np
import random as r

from .Pieces.Bishop import Bishop
from .Pieces.King import King
from .Pieces.Knight import Knight
from .Pieces.Pawn import Pawn
from .Pieces.Queen import Queen
from .Pieces.Rook import Rook


def initialize_pieces(random=False, keep_prob=1.0):

	"""
	Construct list of pieces as objects

	Args: random: Whether board is initialized to random initial state
		  keep_prob: Probability of retaining piece

	Returns: Python list of pieces
	1,1 = a1 ... 8,8 = h8
	"""

	piece_list = [Rook('white',1,1), Knight('white',2,1), Bishop('white',3,1), Queen('white'),
				  King('white'), 	 Bishop('white',6,1), Knight('white',7,1), Rook('white',8,1),
				  Pawn('white',1,2), Pawn('white',2,2),   Pawn('white',3,2),   Pawn('white',4,2),
				  Pawn('white',5,2), Pawn('white',6,2),   Pawn('white',7,2),   Pawn('white',8,2),
				  Pawn('black',1,7), Pawn('black',2,7),   Pawn('black',3,7),   Pawn('black',4,7),
				  Pawn('black',5,7), Pawn('black',6,7),   Pawn('black',7,7),   Pawn('black',8,7),
				  Rook('black',1,8), Knight('black',2,8), Bishop('black',3,8), Queen('black'),
				  King('black'), 	 Bishop('black',6,8), Knight('black',7,8), Rook('black',8,8)]

	# If random is True, randomize piece positions and activity
	if random:
		# For piece in piece list...
		for piece in piece_list:
			# Toggle activity based on uniform distribution (AND PIECE IS NOT KING)
			if r.random() >= keep_prob and piece.name != 'King':
				piece.remove()
			# If the piece was not removed, randomize file and rank
			else:
				newfile = r.randint(1,8)
				newrank = r.randint(1,8)

				# If there is another piece in the target tile, swap places
				for other_piece in piece_list:
					if other_piece.is_active and other_piece.file == newfile and other_piece.rank == newrank:
						# Swap places
						other_piece.file = piece.file
						other_piece.rank = piece.rank
				# Else, and in the previous case, update the piece's file and rank
				piece.file = newfile
				piece.rank = newrank
				piece.move_count += 1


	return piece_list

def board_state(piece_list):

	"""Configuring inputs for value function network

	The output contains M planes of dimensions (N X N) where (N X N) is the size of the board.
	There are M planes "stacked" in layers where each layer represents a different "piece group" 
	(e.g. white pawns, black rooks, etc.) in one-hot format where 1 represents a piece in those
	coordinates and 0 represents the piece is not in those coordinates.

	The M layers each represent a different piece group 
	The order of is as follows:
	- 0: White Pawns 		Pieces 8 - 15
	- 1: White Knights		Pieces 1 and 6
	- 2: White Bishops		Pieces 2 and 5
	- 3: White Rooks 		Pieces 0 and 7
	- 4: White Queen		Piece 3
	- 5: White King 		Piece 4
	- 6: Black Pawns 		Pieces 16 - 23
	- 7: Black Knights 		Pieces 25 and 30
	- 8: Black Bishops 		Pieces 26 and 29
	- 9: Black Rooks 		Pieces 24 and 31
	- 10: Black Queen		Piece 27
	- 11: Black King 		Piece 28
	Note that the number of pieces in each category may change upon piece promotion or removal
	(hence the code below will remain general).
 	"""

	# Define parameters
	N = 8	# N = board dimensions (8 x 8)
	M = 12	# M = piece groups (6 per player)

	# Initializing board state with dimensions N x N x (MT + L)
	board = np.zeros((N,N,M))

	# Fill board state with pieces
	for piece in piece_list:
		if piece.is_active:
			# Place active white pieces in planes 0-5 and continue to next piece
			if piece.color == 'white':
				
				if piece.name == 'Pawn':
					board[piece.file-1, piece.rank-1, 0] = 1

				elif piece.name == 'Knight':
					board[piece.file-1, piece.rank-1, 1] = 1

				elif piece.name == 'Bishop':
					board[piece.file-1, piece.rank-1, 2] = 1

				elif piece.name == 'Rook':
					board[piece.file-1, piece.rank-1, 3] = 1

				elif piece.name == 'Queen':
					board[piece.file-1, piece.rank-1, 4] = 1

				elif piece.name == 'King':
					board[piece.file-1, piece.rank-1, 5] = 1

			# Place active black pieces in planes 6-11 and continue to next piece
			elif piece.color == 'black':

				if piece.name == 'Pawn':
					board[piece.file-1, piece.rank-1, 6] = 1

				elif piece.name == 'Knight':
					board[piece.file-1, piece.rank-1, 7] = 1

				elif piece.name == 'Bishop':
					board[piece.file-1, piece.rank-1, 8] = 1

				elif piece.name == 'Rook':
					board[piece.file-1, piece.rank-1, 9] = 1

				elif piece.name == 'Queen':
					board[piece.file-1, piece.rank-1, 10] = 1

				elif piece.name == 'King':
					board[piece.file-1, piece.rank-1, 11] = 1

	# Return board state
	return board

def visualize_state(piece_list):

	"""
	Visualizing board in terminal

	The output is an 8x8 grid indicating the present locations for each piece
	"""
	# Initializing empty grid
	visualization = np.empty([8,8],dtype=object)
	for i in range(0,8):
		for j in range(0,8):
			visualization[i,j] = ' '

	for piece in piece_list:
		if piece.is_active:
			if piece.color == 'white':
				if piece.name == 'Pawn':
					visualization[piece.file-1, piece.rank-1] = 'P'

				elif piece.name == 'Rook':
					visualization[piece.file-1, piece.rank-1] = 'R'

				elif piece.name == 'Knight':
					visualization[piece.file-1, piece.rank-1] = 'N'

				elif piece.name == 'Bishop':
					visualization[piece.file-1, piece.rank-1] = 'B'
				
				elif piece.name == 'Queen':
					visualization[piece.file-1, piece.rank-1] = 'Q'

				elif piece.name == 'King':
					visualization[piece.file-1, piece.rank-1] = 'K'

			if piece.color == 'black':
				if piece.name == 'Pawn':
					visualization[piece.file-1, piece.rank-1] = 'p'

				elif piece.name == 'Rook':
					visualization[piece.file-1, piece.rank-1] = 'r'

				elif piece.name == 'Knight':
					visualization[piece.file-1, piece.rank-1] = 'n'

				elif piece.name == 'Bishop':
					visualization[piece.file-1, piece.rank-1] = 'b'

				elif piece.name == 'Queen':
					visualization[piece.file-1, piece.rank-1] = 'q'

				elif piece.name == 'King':
					visualization[piece.file-1, piece.rank-1] = 'k'


	# Return visualization
	return visualization


def action_space(piece_list, player):

	"""
	Determining available moves for evaluation

	The output is a P x 56 matrix where P is the numieces and 56ber of p is the maximum
	possible number of moves for any piece. For pieces which have less than  possible
	moves, zeros are appended to the end of the row. A value of 1 indicates that a
	move is available while a value of 0 means that it is not.

	See each pieces in Utils/Pieces/*.py for move glossary

	Return: action space
	"""

	# Initializing action space with dimensions P x 56
	action_space = np.zeros((16,56))

	# For each piece...
	for i in range(0,16):
		# If it is white's turn to move...
		if player == 'white':
			# Obtain vector of possible actions and write to corresponding row
			action_space[i,:] = piece_list[i].actions(piece_list)
		else:
			action_space[i,:] = piece_list[i+16].actions(piece_list)

	# Return action space
	return action_space

def points(piece_list):

	"""
	Calculating point differential for the given board state

	The points are calculated via the standard chess value system:
	Pawn = 1, King = 3, Bishop = 3, Rook = 5, Queen = 9
	King = 100 (arbitrarily large)

	Returns: differential (white points - black points)
	"""

	differential = 0
	# For all white pieces
	for i in range(0,16):
		# If the piece is active, add its points to the counter
		if piece_list[i].is_active:
			differential = differential + piece_list[i].value
	# For all black pieces
	for i in range(16,32):
		# If the piece is active, subtract its points from the counter
		if piece_list[i].is_active:
			differential = differential - piece_list[i].value

	# Return point differential
	return differential
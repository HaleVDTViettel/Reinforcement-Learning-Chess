import keras
import numpy as np
import random as r
import time as t
import copy as c
import os
from Utils import utils as s
import tensorflow as tf

print("Setting memory growth for GPU...")
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("Memory growth set.")

# disable tensorflow logging for clean output
keras.utils.disable_interactive_logging()

# ----------------------------------------------------
# User-Defined Parameters
# ----------------------------------------------------
# Training
NUM_TESTING = 100  # Number of testing steps
HIDDEN_UNITS = 512  # Number of hidden units
BATCH_SIZE = 128  # Batch size

# Simulation
MAX_MOVES = 100  # Maximum number of moves
EPSILON = 0.0  # Defining epsilon for e-greedy policy (0 for testing -> greedy policy)

# Load File
LOAD_FILE = True  # Load trained model from saved checkpoint (True for testing)
VISUALIZE = True  # Select True to visualize games and False to suppress game output
PRINT = True  # Select True to print moves as text and False to suppress printing
ALGEBRAIC = False  # Specify long algebraic notation (True) or descriptive text (False)

# ----------------------------------------------------
# Data Paths
# ----------------------------------------------------
# Specify filenames
# Root directory:
dir_name = "./results/"
load_path = os.path.join(dir_name, "checkpoints/model")  # Model load path
filewriter_path = os.path.join(dir_name, "output")  # Filewriter save path
outcome_file = os.path.join(dir_name, "outcomes.txt")  # Output data filenames (.txt)

# ----------------------------------------------------
# User-Defined Methods
# ----------------------------------------------------
def initialize_board(random=False, keep_prob=1.0):
    """
    Initialize Game Board
    Returns: Game board state parameters
    """
    pieces = s.initialize_pieces(random=random, keep_prob=keep_prob)
    board_state = s.board_state(pieces)
    player = 'white' if not random else 'white' if r.randint(0, 1) == 1 else 'black'
    move = 0
    return pieces, board_state, player, move

def visualize_board(pieces, player, move):
    """
    Visualize Game Board
    Returns: Void
    """
    print("\nCurrent Board at Move " + str(move) + " for Player " + player)
    print(s.visualize_state(pieces))

def move_piece(piece, move_index, player, pieces, switch_player=False, print_move=False, algebraic=True):
    """
    Perform specified move
    Returns: Void
    """
    if player == 'white':
        pieces[piece].move(move_index, pieces, print_move=print_move, algebraic=algebraic)
    else:
        pieces[piece + 16].move(move_index, pieces, print_move=print_move, algebraic=algebraic)
    if switch_player:
        player = 'black' if player == 'white' else 'white'
        return player

def generate_outcome(batch_size, max_moves, epsilon, visualize, print_move, algebraic):
    """
    Generating feature and target batches
    Returns: (1) feature batch, (2) label batch, (3) visualize board?, (4) print move?, (5) print algebraic notation?
    """
    outcome_batch = []

    for batch_step in range(batch_size):
        if visualize or print_move:
            print("\033c", end="")
            # print("Training Game Batch of Size 8192 with Epsilon 0.2")
            print(f"Training Game Batch of Size {batch_size} with Epsilon {epsilon}")
            print(f"-----------BEGIN GAME {batch_step+1}----------")
            
        all_states = []
        all_returns = []
        pieces, initial_state, player, move = initialize_board(random=False, keep_prob=1.0)
        point_diff_0 = s.points(pieces)
        while pieces[4].is_active and pieces[28].is_active and move < max_moves:
            board_state = initial_state if move == 0 else s.board_state(pieces)
            if visualize:
                visualize_board(pieces, player, move)
            net_diff = s.points(pieces) - point_diff_0
            point_diff_0 = s.points(pieces)
            all_states.append(board_state)
            for i in range(len(all_returns)):
                all_returns[i] += net_diff
            all_returns.append(0)
            action_space = s.action_space(pieces, player)
            return_array = np.zeros((16, 56))
            for i in range(16):
                for j in range(56):
                    if action_space[i, j] == 1:
                        temp_pieces = c.deepcopy(pieces)
                        move_piece(i, j, player, temp_pieces)
                        temp_board_state = s.board_state(temp_pieces)
                        for i in range(16):
                            for j in range(56):
                                if action_space[i, j] == 1:
                                    temp_pieces = c.deepcopy(pieces)
                                    move_piece(i, j, player, temp_pieces)
                                    temp_board_state = s.board_state(temp_pieces)
                                    # Predict Q-value for this specific state-action pair
                                    q_value = model.predict([np.reshape(temp_board_state, (1, 768)), 
                                                             np.array([[i*56 + j]])],
                                                             workers=20, use_multiprocessing=True)
                                    return_array[i, j] = q_value[0][0]  # Extract the single Q-value
            if player == 'black':
                while True:
                    piece_index, move_index = r.randint(0, 15), r.randint(0, 55)
                    if return_array[piece_index, move_index] != 0:
                        player = move_piece(piece_index, move_index, player, pieces, switch_player=True, print_move=print_move, algebraic=algebraic)
                        break
            else:
                move_choice = np.nonzero(return_array.max() == return_array)
                piece_index, move_index = move_choice[0][0], move_choice[1][0]
                player = move_piece(piece_index, move_index, player, pieces, switch_player=True, print_move=print_move, algebraic=algebraic)
            move += 1
        if visualize or print_move:
            print(f"----------END OF GAME {batch_step+1}----------")

        if all_returns[0] > 0:
            outcome_batch.append(1)
            
        elif all_returns[0] == 0:
            outcome_batch.append(0)
        else:
            outcome_batch.append(-1)
    return np.array(outcome_batch)

# ----------------------------------------------------
# Importing Session Parameters
# ----------------------------------------------------
state_input = keras.layers.Input(shape=(768,))
action_input = keras.layers.Input(shape=(1,), dtype='int32')

hidden_layer1 = keras.layers.Dense(HIDDEN_UNITS, activation='relu')(state_input)
hidden_layer2 = keras.layers.Dense(HIDDEN_UNITS, activation='relu')(hidden_layer1)
q_values = keras.layers.Dense((16 * 56), activation='linear')(hidden_layer2)

# Use the action input to select the Q-value for the taken action
q_value = keras.layers.Lambda(lambda x: tf.gather(x[0], x[1], batch_dims=1))([q_values, action_input])

model = keras.models.Model(inputs=[state_input, action_input], outputs=q_value)

# ----------------------------------------------------
# Run Session
# ----------------------------------------------------
model.load_weights(load_path)
outcomes = []

start_time = t.time()
for step in range(NUM_TESTING):
    outcome = generate_outcome(batch_size=BATCH_SIZE, max_moves=MAX_MOVES, epsilon=EPSILON, visualize=VISUALIZE, print_move=PRINT, algebraic=ALGEBRAIC)
    outcomes.append(outcome)
    if step % 1 == 0:
        p_completion = 100 * step / NUM_TESTING
        print("\nPercent Completion: %.3f%%" % p_completion)
        avg_elapsed_time = (t.time() - start_time) / (step + 1)
        sec_remaining = avg_elapsed_time * (NUM_TESTING - step)
        min_remaining = round(sec_remaining / 60)
        print("Time Remaining: %d minutes" % min_remaining)
        print(outcome)
        print("Mean Outcome: %.3f" % np.mean(outcomes))

outcomes = np.array(outcomes)
with open(outcome_file, 'a') as file_object:
    np.savetxt(file_object, outcomes)

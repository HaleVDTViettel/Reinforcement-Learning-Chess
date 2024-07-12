
import numpy as np
import random as r
import time as t
import copy as c
import os
from Utils import utils as s
from Utils import sysinfo as si
import torch
import torch.nn as nn 
import torch.nn.functional as F

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # Initialize board pieces
    pieces = s.initialize_pieces(random=random, keep_prob=keep_prob)
    # Initialize state space 
    board_state = s.board_state(pieces)
    # Initialize current player:
    player = 'white' if not random or r.randint(0, 1) == 1 else 'black'
    # Initialize move counter:
    move = 0

    # Return values
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
    """
    if player == 'white':
        pieces[piece].move(move_index, pieces, print_move=print_move, algebraic=algebraic)
    else:
        pieces[piece+16].move(move_index, pieces, print_move=print_move, algebraic=algebraic)

    if switch_player:
        player = 'black' if player == 'white' else 'white'
        return player

def generate_outcome(model, batch_size, max_moves, epsilon, visualize, print_move, algebraic):
    # print("Generating game...\r\n")
    # Generates training data based on batches of full-depth Monte-Carlo simulations
    # performing epsilon-greedy policy evaluation.
    move_batches = []
    # update the time running sequentially 
    # Loop through batch steps
    for batch_step in range(0, batch_size):
        if visualize or print_move:
            print("\033c", end="")
            # print("Training Game Batch of Size 8192 with Epsilon 0.2")
            print(f"Training Game Batch of Size {batch_size} with Epsilon {epsilon}")
            print(f"-----------BEGIN GAME {batch_step+1}----------")
            si.print_system_info()
            print(f"_"*50)

        # ----------------------------------------------------
        # Initialize Board State
        # ----------------------------------------------------
        # placeholders for board states and return for each state
        all_states = []
        all_returns = []

        # Generating board parameters
        pieces, initial_state, player, move = initialize_board(random=False, keep_prob=0.65)
        point_diff_0 = s.points(pieces)


        # ----------------------------------------------------
        # Monte Carlo Simulations
        # ----------------------------------------------------

        # Terminal events: Kings.is_active == False or move_counter > MAX_MOVES
        while pieces[4].is_active and pieces[28].is_active and move < max_moves:

            # Obtain board state
            board_state = initial_state if move == 0 else s.board_state(pieces)

            # Visualize board state
            if visualize:
                visualize_board(pieces, player, move)

            # Obtain current point differential
            net_diff = s.points(pieces) - point_diff_0
            point_diff_0 = s.points(pieces)
            
            # Append initial board state to all_states
            all_states.append(board_state)
            # Add net_diff to all existing returns
            for i in range(0, len(all_returns)):
                all_returns[i] += net_diff
            # Append 0 to end of all_returns representing return for current state
            all_returns.append(0)

            # Obtain action space
            action_space = s.action_space(pieces, player)


            # ----------------------------------------------------
            # Value Function Approximation
            # ----------------------------------------------------
            # For each action in the action space, obtain subsequent board space
            # and calculate estimated return with the partially-trained approximator
            return_array = np.zeros((16, 56))

            # For each possible move...
            for i in range(0, 16):
                for j in range(0, 56):
                    # If the move is legal...
                    if action_space[i, j] == 1:

                        # Perform move and obtain temporary board state
                        temp_pieces = c.deepcopy(pieces)                 # Reset temporary pieces variable
                        move_piece(i, j, player, temp_pieces)            # Perform temporary move
                        temp_board_state = s.board_state(temp_pieces)    # Obtain temporary state

                        # Convert to tensor and move to correct device
                        temp_state = torch.FloatTensor(temp_board_state).to(device)
                        with torch.no_grad():
                            expected_return = model(temp_state.unsqueeze(0)).squeeze(0).max().item()
                        return_array[i, j] = expected_return


            # ----------------------------------------------------
            # Policy
            # ----------------------------------------------------
        
            # For player black, choose a random action
            if player == 'black':
                while True:
					# If the action is valid...
                    piece_index = r.randint(0,15)
                    move_index = r.randint(0,55)
                    if return_array[piece_index,move_index] != 0:
						# Perform move and update player
                        player = move_piece(piece_index,move_index,player,pieces,switch_player=True,print_move=print_move,algebraic=algebraic)
                        break
            else:
                # Identify indices of maximum return (white) or minimum return (black)
                move_choice = np.nonzero(return_array.max() == return_array)
                piece_index = move_choice[0][0]
                move_index = move_choice[1][0]
				# Perform move and update player
                # Perform move and update player
                player = move_piece(piece_index, move_index, player, pieces, switch_player=True, print_move=print_move, algebraic=algebraic)
            # Increment move counter
            move += 1

        if visualize or print_move:
            print(f"----------END OF GAME {batch_step+1}----------")
            pass

        # If player white won the game...
        if all_returns[0] > 0: 
            move_batches.append(1)		# Return 1
		# Else, for a draw...
        elif all_returns[0] == 0:
            move_batches.append(0)		# Return 0 
		# Else, if player black won the game...
        else:
            move_batches.append(-1)	# Return -1
    outcome_batchs = torch.FloatTensor(np.array(move_batches))
    return outcome_batchs

# ----------------------------------------------------
# Importing Session Parameters
# ----------------------------------------------------
class DQNModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_units):
        super(DQNModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units

        # board state_size is (8, 8, 12)
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, action_size)

    def forward(self, state, action=None):
        # Reshape state to (batch_size, channels, height, width)
        x = state.view(-1, 12, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        if action is not None:
            # If action is provided, return the Q-value for the taken action
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            return q_value
        else:
            # If no action is provided, return all Q-values
            return q_values
        
    def load_weights(self, weights):
        torch.load(weights)

state_size = 768  # 8x8x12
action_size = 16 * 56  # 16 pieces, 56 possible moves each

model = DQNModel(state_size, action_size, HIDDEN_UNITS).to(device)
load_path = "./results/checkpoints/model.pth"
model.load_weights(load_path)
# ----------------------------------------------------
# Run Session
# ----------------------------------------------------
outcomes = []

start_time = t.time()
for step in range(NUM_TESTING):
    outcome = generate_outcome(model, batch_size=BATCH_SIZE, max_moves=MAX_MOVES, epsilon=EPSILON, visualize=VISUALIZE, print_move=PRINT, algebraic=ALGEBRAIC)
    outcomes.append(outcome)
    if step % 1 == 0:
        p_completion = 100 * step / NUM_TESTING
        print("\nPercent Completion: %.3f%%" % p_completion)
        avg_elapsed_time = (t.time() - start_time) / (step + 1)
        sec_remaining = avg_elapsed_time * (NUM_TESTING - step)
        min_remaining = round(sec_remaining / 60)
        print("Time Remaining: %d minutes" % min_remaining)
        print(outcome)
        print("Mean Outcome: %.3f" % torch.mean(torch.stack(outcomes)).item())

outcomes = np.array(outcomes)
with open(outcome_file, 'a') as file_object:
    np.savetxt(file_object, outcomes)

import warnings
warnings.filterwarnings("ignore")

import os
import time
import keras
import argparse
import time as t
import copy as c
import threading
import numpy as np
import random as r
import tensorflow as tf
from collections import deque
from keras.optimizers import AdamW

from Utils import utils as s

# disable tensorflow logging for clean output
keras.utils.disable_interactive_logging()

# set memory growth for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Function to print the elapsed time continuously
def print_time():
    start_time = time.time()
    while not stop_thread:
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds", end='\r')
        time.sleep(0.1)

def print_loss(loss):
    print(f"Loss: {loss:.4f}", end='\r')

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_units, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        state_input = keras.layers.Input(shape=(self.state_size,))
        action_input = keras.layers.Input(shape=(1,), dtype='int32')
        
        hidden_layer1 = keras.layers.Dense(self.hidden_units, activation='relu')(state_input)
        hidden_layer2 = keras.layers.Dense(self.hidden_units, activation='relu')(hidden_layer1)
        q_values = keras.layers.Dense(self.action_size, activation='linear')(hidden_layer2)

        # Use the action input to select the Q-value for the taken action
        q_value = keras.layers.Lambda(lambda x: tf.gather(x[0], x[1], batch_dims=1))([q_values, action_input])

        model = keras.models.Model(inputs=[state_input, action_input], outputs=q_value)

        model.compile(optimizer=keras.optimizers.AdamW(learning_rate=self.learning_rate, 
                                                    beta_1=0.9, 
                                                    beta_2=0.999, 
                                                    epsilon=1e-07, 
                                                    weight_decay=0.01, 
                                                    amsgrad=True),
                    loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return r.randrange(self.action_size), None
        
        state = np.reshape(state, (1, self.state_size))
        action_input = np.array([[0]])
        q_values = self.model.predict([state, action_input])
        action = np.argmax(q_values)
        
        return action, q_values.flatten()    

    def replay(self, batch_size):
        minibatch = r.sample(self.memory, batch_size)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        targets = rewards
        
        history = self.model.fit([states, actions], targets, epochs=1, verbose=0, use_multiprocessing=True, workers=20)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def flatten_state(board_state):
    return board_state.flatten()

def generate_game_step(agent, pieces, player, move, visualize=False, print_move=False, algebraic=False):
    board_state = s.board_state(pieces)
    state = flatten_state(board_state)
    
    action_space = s.action_space(pieces, player)
    action, _ = agent.act(state)  # state is already flattened
    
    piece_index = action // 56
    move_index = action % 56
    
    if action_space[piece_index, move_index] == 1:
        player = move_piece(piece_index, move_index, player, pieces, switch_player=True, print_move=print_move, algebraic=algebraic)
        move += 1
        next_board_state = s.board_state(pieces)
        next_state = flatten_state(next_board_state)
        reward = s.points(pieces)  # You might want to design a better reward function
        done = not (pieces[4].is_active and pieces[28].is_active) or move >= max_moves
        
        if visualize:
            visualize_board(pieces, player, move)
        
        return next_state, reward, done, player, move, action
    else:
        # Invalid move, penalize
        return state, -10, False, player, move, action

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

def generate_game(agent, batch_size, max_moves, epsilon, visualize, print_move, algebraic):
    """
    Generating feature and target batches
    """

    # Generates training data based on batches of full-depth Monte-Carlo simulations
    # performing epsilon-greedy policy evaluation.

    feature_batches = []
    label_batches = []
    print(f"Training Game Batch of Size {batch_size} with Epsilon {epsilon}")
    # update the time running sequentially 
    # Loop through batch steps
    for batch_step in range(0, batch_size):
        if visualize or print_move:
            print(f"\n-----------BEGIN GAME {batch_step}----------")

        # ----------------------------------------------------
        # Initialize Board State
        # ----------------------------------------------------
        # Create placeholders for board states and return for each state
        all_states = []
        all_returns = []

        # Generating board parameters
        pieces, initial_state, player, move = initialize_board(random=True, keep_prob=0.85)
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
                        temp_pieces = c.deepcopy(pieces)                # Reset temporary pieces variable
                        move_piece(i, j, player, temp_pieces)                # Perform temporary move
                        temp_board_state = s.board_state(temp_pieces)    # Obtain temporary state

                        # With temporary state, calculate expected return
                        expected_return = agent.model.predict([np.reshape(temp_board_state, (1, 768)), np.array([[0]])])
                        # Estimated return to return_array
                        return_array[i, j] = expected_return


            # ----------------------------------------------------
            # Epsilon-Greedy Policy
            # ----------------------------------------------------

            if r.random() < epsilon:
                while True:
                    # If the action is valid...
                    piece_index = r.randint(0, 15)
                    move_index = r.randint(0, 55)
                    if return_array[piece_index, move_index] != 0:
                        # Perform move and update player
                        player = move_piece(piece_index, move_index, player, pieces, switch_player=True, print_move=print_move, algebraic=algebraic)
                        break

            else:
                # Identify indices of maximum return (white) or minimum return (black)
                if player == 'white':
                    # Find the indices of the maximum nonzero value
                    maxval = np.max(return_array[np.nonzero(return_array)])
                    maxdim = np.argwhere(return_array == maxval)
                    piece_index = maxdim[0][0]    # Maximum (row)
                    move_index = maxdim[0][1]    # Maximum (column)
                else:
                    # Find the indices of the minimum nonzero value
                    minval = np.min(return_array[np.nonzero(return_array)])
                    mindim = np.argwhere(return_array == minval)
                    piece_index = mindim[0][0]    # Maximum (row)
                    move_index = mindim[0][1]    # Maximum (column)
                # Perform move and update player
                player = move_piece(piece_index, move_index, player, pieces, switch_player=True, print_move=print_move, algebraic=algebraic)
            # Increment move counter
            move += 1

        if visualize or print_move:
            print(f"----------END OF GAME {batch_step}----------")

        feature_batches.append(initial_state)
        label_batches.append(all_returns[0])

    # Return features and labels
    feature_batches = np.array(feature_batches)
    label_batches = np.array(label_batches)
    return feature_batches, label_batches

class Callback(tf.keras.callbacks.Callback):
    SHOW_NUMBER = 10
    counter = 0
    epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if self.counter == self.SHOW_NUMBER or self.epoch == 1:
            print('Epoch: ' + str(self.epoch) + ' loss: ' + str(logs['loss']))
            if self.epoch > 1:
                self.counter = 0
        self.counter += 1


if __name__ == "__main__":

    # ----------------------------------------------------
    # Parsing Console Arguments
    # ----------------------------------------------------
    
    # Create parser object
    parser = argparse.ArgumentParser()

    parser.add_argument("-t",  "--trainsteps",   help="Number of training steps (Default 1000)",             type=int,   default=1000)
    parser.add_argument("-u",  "--hidunits",     help="Number of hidden units (Default 100)",                type=int,   default=100)
    parser.add_argument("-r",  "--learnrate",    help="Learning rate (Default 0.001)",                       type=float, default=0.001)
    parser.add_argument("-b",  "--batchsize",    help="Batch size (Default 32)",                             type=int,   default=32)
    parser.add_argument("-m",  "--maxmoves",     help="Maximum moves for MC simulations (Default 100)",      type=int,   default=100)
    parser.add_argument("-e",  "--epsilon",      help="Epsilon-greedy policy evaluation (Default 0.2)",      type=float, default=0.2)
    parser.add_argument("-v",  "--visualize",    help="Visualize game board? (Default False)",               type=bool,  default=False)
    parser.add_argument("-p",  "--print",        help="Print moves? (Default False)",                        type=bool,  default=False)
    parser.add_argument("-a",  "--algebraic",    help="Print moves in algebraic notation? (Default False)",  type=bool,  default=False)
    parser.add_argument("-l",  "--loadfile",     help="Load model from saved checkpoint? (Default False)",   type=bool,  default=False)
    parser.add_argument("-rd", "--rootdir",      help="Root directory for project",                          type=str,   default="./results")
    parser.add_argument("-sd", "--savedir",      help="Save directory for project",                          type=str,   default="checkpoints/model")
    parser.add_argument("-ld", "--loaddir",      help="Load directory for project",                          type=str,   default="checkpoints/model")
    parser.add_argument("-w",  "--workers",      help="Number of workers for parallel processing",           type=int,   default=os.cpu_count())

    # Parse Arguments from Command Line
    args = parser.parse_args()

    # Value Function Approximator Training
    num_training = args.trainsteps
    hidden_units = args.hidunits
    learning_rate = args.learnrate
    batch_size = args.batchsize
    workers = args.workers

    # Simulation Parameters
    max_moves = args.maxmoves
    epsilon = args.epsilon
    visualize = args.visualize
    print_moves = args.print
    algebraic = args.algebraic

    # Load File
    load_file = args.loadfile

    # File Paths
    dir_name = args.rootdir
    load_path = args.loaddir
    save_path = args.savedir

    
    filewriter_path = os.path.join(dir_name, "output")                    # Filewriter save path
    training_loss = os.path.join(dir_name, "training_loss.txt")            # Training loss

    tf.config.threading.set_inter_op_parallelism_threads(workers) 
    tf.config.threading.set_intra_op_parallelism_threads(workers)

    # parse example: python main.py -t 1000 -u 512 -r 0.001 -b 4096 -m 3100 -e 0.2 -v True -p True -a True -l False -rd ./results -sd checkpoints/model -ld checkpoints/model

    """
    Train value function model

    Arguments:
    - num_training:        [int]            Number of training steps
    - hidden_units:        [int]            Number of hidden units per layer
    - learning_rate:       [float]          Initial learning rate
    - batch_size:          [int]            Batch size for stochastic gradient descent
    - max_moves:           [int]            Maximum moves for Monte Carlo simulations
    - epsilon:             [float]          Epsilon-greedy policy parameter
    - visualize:           [bool]           Visualize game board during training?
    - print_moves:         [bool]           Print moves during training?
    - algebraic:           [bool]           Print moves using algebraic notation or long-form?
    - load_file:           [bool]           Load pre-trained model?
    - dir_name:            [str]            Root directory filepath
    - load_path:           [str]            Path to pre-trained model from root directory
    - save_path:           [str]            Save path from root directory
    - filewriter_path:     [str]            Save path for filewriter (TensorBoard)
    - training_loss        [str]            Output .txt file name / path for training loss
    """

    state_size = 768  # 8x8x12
    action_size = 16 * 56  # 16 pieces, 56 possible moves each
    agent = DQNAgent(state_size, action_size, hidden_units, learning_rate)

    if load_file:
        agent.load(load_path)

    start_time = t.time()
    # Variable to control the timer thread
    stop_thread = False

    for episode in range(num_training):
        # Start the timer thread
        timer_thread = threading.Thread(target=print_time)
        timer_thread.start()
        # Initialize training loss
        t_loss = []
        # Generate a batch of training data
        feature_batch, label_batch = generate_game(agent, batch_size, max_moves, epsilon, visualize, print_moves, algebraic)

        # Train the agent on this batch
        loss = 0
        for state, target_value in zip(feature_batch, label_batch):
            state = flatten_state(state)
            action, _ = agent.act(state)
            agent.remember(state, action, target_value, state, False)  # Use the same state for next_state as we don't have it
            if len(agent.memory) > batch_size:
                loss += agent.replay(batch_size)

        # Stop the timer thread
        stop_thread = True
        timer_thread.join()  # Wait for the timer thread to finish

        # Clear the printed line
        print("\r" + " " * 40 + "\r", end='')

        # Average loss over the batch
        avg_loss = loss / batch_size if batch_size > 0 else 0
        t_loss.append(avg_loss)

        if episode % 10 == 0:
            agent.update_target_model()
            agent.save(save_path)

            # Report progress
            print(f"\nEpisode: {episode}/{num_training}, Epsilon: {agent.epsilon:.2f}")
            print(f"Average train loss: {avg_loss:.4f}")

            # Report percent completion and time remaining
            p_completion = 100 * episode / num_training
            print("\nPercent completion: %.3f%%" % p_completion)

            avg_elapsed_time = (t.time() - start_time) / (episode + 1)
            sec_remaining = avg_elapsed_time * (num_training - episode)
            min_remaining = round(sec_remaining / 60)
            print("Time Remaining: %d minutes" % min_remaining)


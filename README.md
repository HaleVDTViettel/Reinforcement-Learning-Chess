# REL301m Reinforcement Learning Assignment - Chess

This project is a reinforcement learning assignment for the course REL301m at the University of FPTUniversity HCM campus. The goal of the assignment is to create an engine that uses reinforcement learning from scratch to play chess. The engine will be trained using self-play and will be evaluated against other engines.

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)

## About <a name = "about"></a>

The action-value functions are developed by training a neural network on the overall returns of board states initialized randomly, as determined by [Monte Carlo simulations](https://www.youtube.com/watch?v=7TqhmX92P6U). The system adheres to an [epsilon-greedy policy](https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/) guided by the latest approximations of the action-value functions. Starting from version, each training step utilizes batches derived from comprehensive Monte Carlo simulations. The model architecture includes two hidden layers, but it can be easily expanded or modified to a convolutional architecture, which is planned for a future update.

## Getting Started <a name = "getting_started"></a>

Folk this repository and clone it to your local machine. You can use the following command to clone the repository.

```bash
gh repo clone HaleVDTViettel/Reinforcement-Learning-Chess
```


### Prerequisites
This project requires Python 3.8 or later and TensorFlow with GPU support 2.15.1 or later.

Install the required packages by running the following command.

```bash
pip install -r requirements.txt
```
>Note: I will make a docker image for this project soon.

## Usage <a name = "usage"></a>

To train the model, run the following command with default settings.

```bash
python3 main.py
```
For more information, please use the help command.

```bash
python3 main.py --help
```
Example command:

```bash
python3 main.py -t 1000 \
               -u 512 \
               -r 0.001 \
               -b 4096 \
               -m 3100 \
               -e 0.2 \
               -v True \
               -p True \
               -a True \
               -l False \
               -rd ./results \
               -sd checkpoints/model \
               -ld checkpoints/model
```
Or you can try the following command.

```bash
python3 main.py [-h] [-t TRAINSTEPS] [-u HIDUNITS] [-r LEARNRATE]
               [-b BATCHSIZE] [-m MAXMOVES] [-e EPSILON] [-v VISUALIZE]
               [-p PRINT] [-a ALGEBRAIC] [-l LOADFILE] [-rd ROOTDIR]
               [-sd SAVEDIR] [-ld LOADDIR]
```

### Arguments:

```console
  -h,               --help   show this help message and exit
  -t TRAINSTEPS,    --trainsteps TRAINSTEPS
                        Number of training steps (Default 1000)
  -u HIDUNITS,      --hidunits HIDUNITS
                        Number of hidden units (Default 100)
  -r LEARNRATE,     --learnrate LEARNRATE
                        Learning rate (Default 0.001)
  -b BATCHSIZE,     --batchsize BATCHSIZE
                        Batch size (Default 32)
  -m MAXMOVES,      --maxmoves MAXMOVES
                        Maximum moves for MC simulations (Default 100)
  -e EPSILON,       --epsilon EPSILON
                        Epsilon-greedy policy evaluation (Default 0.2)
  -v VISUALIZE,     --visualize VISUALIZE
                        Visualize game board? (Default False)
  -p PRINT,         --print PRINT
                        Print moves? (Default False)
  -a ALGEBRAIC,     --algebraic ALGEBRAIC
                        Print moves in algebraic notation? (Default False)
  -l LOADFILE,      --loadfile LOADFILE
                        Load model from saved checkpoint? (Default False)
  -rd ROOTDIR,      --rootdir ROOTDIR
                        Root directory for project
  -sd SAVEDIR,      --savedir SAVEDIR
                        Save directory for project
  -ld LOADDIR,      --loaddir LOADDIR
                        Load directory for project
 ```
### Optional
Run **`test_agent.py`** to compare model performance against a benchmark. At the current time, the benchmark is a random policy.
    
```bash
python3 test_agent.py
```

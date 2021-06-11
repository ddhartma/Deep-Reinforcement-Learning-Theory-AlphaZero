[image1]: assets/1.png 
[image2]: assets/2.png 
[image3]: assets/3.png 
[image4]: assets/4.png 
[image5]: assets/5.png 
[image6]: assets/6.png 
[image7]: assets/7.png 
[image8]: assets/8.png 
[image9]: assets/9.png 
[image10]: assets/10.png 
[image11]: assets/11.png 
[image12]: assets/12.png 
[image13]: assets/13.png 
[image14]: assets/14.png 
[image15]: assets/15.png 
[image16]: assets/16.png 
[image17]: assets/17.png 
[image18]: assets/18.png 


# Deep Reinforcement Learning Theory - AlphaZero

Let's introduce the concept of Alpha Zero

## Content 
- [Introduction](#intro)
- [Introduction of AlphaZero](#intro_alpha)
- [Zero-Sum Game](#zero_sum_game)
- [Monte Carlo Tree Search 1 - Random Sampling](#tree_search_1)
- [Monte Carlo Tree Search 2 - Expansion and Back-propagation](#tree_search_2)
- [AlphaZero 1: Guided Tree Search](#guided_tree_search)
- [AlphaZero 2: Self-Play Training](#self_playing)
- [TicTacToe using AlphaZero - code](#tic_tac_toe_code)
- [Advanced TicTacToe with AlphaZero - Walkthrough](#tic_tac_toe_adv)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a name="intro"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

### Overview:
- ***Policy-Based Methods***: methods such as 
    - hill climbing
    - simulated annealing
    - adaptive noise scaling
    - cross-entropy methods
    - evolution strategies

- ***Policy Gradient Methods***:
    - REINFORCE
    - lower the variance of policy gradient algorithms

- ***Proximal Policy Optimization***:
    - Proximal Policy Optimization (PPO), a cutting-edge policy gradient method

- ***Actor-Critic Methods***
    - how to combine value-based and policy-based methods
    - bringing together the best of both worlds, to solve challenging reinforcement learning problems

## Introduction of AlphaZero <a name="intro_alpha"></a> 
In **2016**: researchers at DeepMind  introduced **new AI engine, AlphaGo** for the game of Go. The AI was able to beat a professional player LeeSedol. The breakthrough was significant, because go was far more complex than chess: the number of possible games is so high, that a professional go engine was believed to be way out of reach at that point, and human intuition was believed to be a key component in professional play. Still, performance in Alphago depends on **expert input** during the training step, and so the algorithm cannot be easily be transferred to other domains.

This changed in **2017**, when the team at DeepMind updated their algorithm, and developed a **new engine called AlphaGo Zero**. This time, instead of depending on expert gameplay for the training, AlphaGo Zero **learned from playing against itself**, only knowing the rules of the game. More impressively, the algorithm was generic enough to be adapted to **chess** and **shogi** (also known as japanese chess). This leads to an entirely new framework for developing AI engines, and the researchers called their algorithm, simply as the **AlphaZero**.

The best part of the AlphaZero algorithm is **simplicity**: 
    - it consists of a **Monte Carlo tree search**, 
    - guided by a **deep neural network**. 

This is analogous to the way humans think about board games -- where professional players employ **hard calculations guides with intuitions**.

- Paper [2017, Silver et al. Mastering the Game of Go without Human Knowledge --> AlphaGo Zero](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)
- Paper [2017, Silver et al., Mastering Chess and Shogi by Self-Play with aGeneral Reinforcement Learning Algorithm --> AlphaZero](https://arxiv.org/pdf/1712.01815.pdf)

## Zero-Sum Game <a name="zero_sum_game"></a>  
- **AlphaZero** is specialzed in so called **Zero-Sum games**.
- Assumption: game contains no hidden information (no element of luck, winning or losing is entirely determined by skill)
- This concept is applicable to games as simple as Tic-Tac-Toe to more complicated games such as chess and go.
- Exmaple: Tic-Tac-Toe
    - Grid
    - 2 competing agents 
    - One agent's win is another agent's loss.
    - Goal is to get three in a row,
    - The board can be represented by a matrix 
        - 0 indicates empty space
        - +1 indicates the pieces of player one
        - -1 indicates the pieces of player two
    - Final outcome can be encoded by a score,
        - +1 indicates a win by the first player,
        - -1 indicates a win by the second player,
        - 0 indicates a draw.

- Such a board can easily be fed into a neural network.
- In order to switch players, multiply the matrix by -1.
- Flip the score by multiplying it by -1.
- This property will come in handy when we build an agent to play the game.

    ![image1]

### Rephrase everything in the language of reinforcement learning:
- Sequence of states for the board game denoted by timestep **t**
- Two players denoted by **+1** or **-1** (formula: **-1<sup>t</sup>**,
- Player **+1** acts at all **even timesteps** and tries to maximize the final score plus +z,
- Player **-1** aczs at all **odd timesteps** and tries to maximize final score -z.
- **Common policy**:
    - One intelligent agent who can play both players by flipping the state **s<sub>t</sub>, at all the odd timesteps.
- **Common critic**:
    - One common critic that can evaluate the expected outcome from the perspective of the current player.
    - This essentially estimates the expected value **(-1)<sup>t</sup>z**

    ![image2]

- This is the basic idea behind AlphaZero, where we have one agent playing against itself along with one critic that self-improves as more and more games are played.

## Monte Carlo Tree Search 1 - Random Sampling <a name="Setup_Instructions"></a> tree_search_1)

## Monte Carlo Tree Search 2 - Expansion and Back-propagation <a name="tree_search_2"></a> 

## AlphaZero 1: Guided Tree Search <a name="guided_tree_search"></a> 

## AlphaZero 2: Self-Play Training <a name="self_playing"></a> 

## TicTacToe using AlphaZero - code <a name="tic_tac_toe_code"></a> 

## Advanced TicTacToe with AlphaZero - Walkthrough <a name="tic_tac_toe_adv"></a> 


## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Deep-Reinforcement-Learning-Theory-AlphaZero.git
```

- Change Directory
```
$ cd Deep-Reinforcement-Learning-Theory-AlphaZero
```

- Create a new Python environment, e.g. alpha_zero. Inside Git Bash (Terminal) write:
```
$ conda create --name alpha_zero
```

- Activate the installed environment via
```
$ conda activate alpha_zero
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Deep Reinforcement Learning'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Very good summary of DQN](https://medium.com/@nisheed/udacity-deep-reinforcement-learning-project-1-navigation-d16b43793af5)
* [An Introduction to Deep Reinforcement Learning](https://thomassimonini.medium.com/an-introduction-to-deep-reinforcement-learning-17a565999c0c)
* Helpful medium blog post on policies [Off-policy vs On-Policy vs Offline Reinforcement Learning Demystified!](https://kowshikchilamkurthy.medium.com/off-policy-vs-on-policy-vs-offline-reinforcement-learning-demystified-f7f87e275b48)
* [Understanding Baseline Techniques for REINFORCE](https://medium.com/@fork.tree.ai/understanding-baseline-techniques-for-reinforce-53a1e2279b57)
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Cheat Sheet](https://towardsdatascience.com/reinforcement-learning-cheat-sheet-2f9453df7651)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)

Important publications
* [2004 Y. Ng et al., Autonomoushelicopterflightviareinforcementlearning --> Inverse Reinforcement Learning](https://people.eecs.berkeley.edu/~jordan/papers/ng-etal03.pdf)
* [2004 Kohl et al., Policy Gradient Reinforcement Learning for FastQuadrupedal Locomotion --> Policy Gradient Methods](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/icra04.pdf)
* [2013-2015, Mnih et al. Human-level control through deep reinforcementlearning --> DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [2014, Silver et al., Deterministic Policy Gradient Algorithms --> DPG](http://proceedings.mlr.press/v32/silver14.html)
* [2015, Lillicrap et al., Continuous control with deep reinforcement learning --> DDPG](https://arxiv.org/abs/1509.02971)
* [2015, Schulman et al, High-Dimensional Continuous Control Using Generalized Advantage Estimation --> GAE](https://arxiv.org/abs/1506.02438)
* [2016, Schulman et al., Benchmarking Deep Reinforcement Learning for Continuous Control --> TRPO and GAE](https://arxiv.org/abs/1604.06778)
* [2017, PPO](https://openai.com/blog/openai-baselines-ppo/)
* [2018, Bart-Maron et al., Distributed Distributional Deterministic Policy Gradients](https://openreview.net/forum?id=SyZipzbCb)
* [2013, Sergey et al., Guided Policy Search --> GPS](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf)
* [2015, van Hasselt et al., Deep Reinforcement Learning with Double Q-learning --> DDQN](https://arxiv.org/abs/1509.06461)
* [1993, Truhn et al., Issues in Using Function Approximation for Reinforcement Learning](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)
* [2015, Schaul et al., Prioritized Experience Replay --> PER](https://arxiv.org/abs/1511.05952)
* [2015, Wang et al., Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [2016, Silver et al., Mastering the game of Go with deep neural networks and tree search](https://www.researchgate.net/publication/292074166_Mastering_the_game_of_Go_with_deep_neural_networks_and_tree_search)
* [2017, Hessel et al. Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
* [2016, Mnih et al., Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
* [2017, Bellemare et al., A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
* [2017, Fortunato et al., Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
* [2016, Wang et al., Sample Efficient Actor-Critic with Experience Replay --> ACER](https://arxiv.org/abs/1611.01224)
* [2017, Lowe et al. Multi-Agent Actor-Critic for MixedCooperative-Competitive Environments](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)
* [2017, Silver et al. Mastering the Game of Go without Human Knowledge --> AlphaGo Zero](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)
* [2017, Silver et al., Mastering Chess and Shogi by Self-Play with aGeneral Reinforcement Learning Algorithm --> AlphaZero](https://arxiv.org/pdf/1712.01815.pdf)

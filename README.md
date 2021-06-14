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
In **2016**: researchers at DeepMind  introduced **new AI engine, AlphaGo** for the game of Go. The AI was able to beat a professional player Lee Sedol. The breakthrough was significant, because Go was far more complex than chess: the number of possible games is so high, that a professional go engine was believed to be way out of reach at that point, and human intuition was believed to be a key component in professional play. Still, performance in Alphago depends on **expert input** during the training step, and so the algorithm cannot be easily be transferred to other domains.

This changed in **2017**, when the team at DeepMind updated their algorithm, and developed a **new engine called AlphaGo Zero**. This time, instead of depending on expert gameplay for the training, AlphaGo Zero **learned from playing against itself**, only knowing the rules of the game. More impressively, the algorithm was generic enough to be adapted to **chess** and **shogi** (also known as japanese chess). This leads to an entirely new framework for developing AI engines, and the researchers called their algorithm, simply as the **AlphaZero**.

The best part of the AlphaZero algorithm is **simplicity**: 
    - it consists of a **Monte Carlo tree search**, 
    - guided by a **deep neural network**. 

This is analogous to the way humans think about board games -- where professional players combine **hard calculations with intuition**.

- Paper [2017, Silver et al. Mastering the Game of Go without Human Knowledge --> AlphaGo Zero](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)
- Paper [2017, Silver et al., Mastering Chess and Shogi by Self-Play with aGeneral Reinforcement Learning Algorithm --> AlphaZero](https://arxiv.org/pdf/1712.01815.pdf)

## Zero-Sum Game <a name="zero_sum_game"></a>  
- **AlphaZero** is specialzed in so called **Zero-Sum games**.
- Assumption: game contains no hidden information (no element of luck, winning or losing is entirely determined by skill)
- This concept is applicable to games as simple as Tic-Tac-Toe to more complicated games such as chess and Go.
- Example: Tic-Tac-Toe
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
- Two players denoted by **+1** or **-1** --> formula: **(-1)<sup>t</sup>**,
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
Given a **state in a zero sum game**, how do we find an **optimal policy**?

### In theory: Brute Force
-  Perform **brute force search**, going through all the possible moves and all the possible games that can be played, and then we can choose the ones with the best possible outcomes.
- Problem: to many possibilities, brute force method **can become infeasible**.

### Optimization 1: Random sampling
- **Sample randomly a subset** of all the possible games.
- Then for each action, compute the **expected outcome** by taking the **average of all the subsequent playouts** like this.
- Add an extra **negative sign** so that the expected outcome is from the perspective of the current player.
- Choose the **action with the largest expected score**, in this case **who plays across at the bottom corner**.

    ![image3]

### Optimization 2: Tree structure
- This procedure sounds a little bit familiar: Trying to **compute the expected outcome** is analogous to **estimating the Q function**,
- If we would have some guidance instead of full randomness we could better balance **random explorations with exploitations**.
- Let's get systematic then, and look at all the possible moves **player 2** can take given the current state. There are six possibilities.
- We can think of all the possibilities as part of a **tree structure**.
- The goal is we want to **look at each branch of this** tree and focus on the **actions** that are more
**promising** (vetter than just sample completely randomly).
    - To do this we define three numbers for each branch of this tree: 
        - **U**: number built up from amount of explorations (determined by N: branches with low visit counts increases U and are favored) and exploitation (determined by V)
        - **N**: number of times visiting branch
        - **V**: estimate of the expected outcome
    - All the values are initialized to zero.
    - In each iteration play a random game starting from a selective branch.
    - The branch with the highest value of **U** will be chosen.
- Example game:
    - Initially, all the U's are zero, so we just randomly choose one of the six possible actions.
    - Play a random game, and this time player 2 lost.
    - So, V is updated to be **V=-1**, and the number of visit increased to **N=1**. 
    - Total number of gameplay is updated to **N<sub>tot</sub>=1**.
    - The **exploration part of the U function needs to be updated** for all the other branches.
    - The largest U is now **U=1**, so going to the next iteration of gameplay, we need to choose one of the other five actions.
    - We can then repeat this procedure until a **fixed number of total gameplays**, say 100 games is reached, and we might end up with something like this.
    - We could choose the action with the highest U or V, but instead, we **choose an action with the highest visit counts**, which is usually associated with **good expected outcome** anyway, and it's a little bit **more reliable**.
    - In this case the highest visit count is 21.
    - This is the **move that we will choose** according to the **tree search**.

    ![image4]

## Monte Carlo Tree Search 2 - Expansion and Back-propagation <a name="tree_search_2"></a> 
Can we generalize the Tree Search concept to go deeper into the tree so that we can better anticipate a long sequence of moves?
### Expansion and backpropagation
- **Let's see it in action**: 
    - First, we choose an **action that maximizes U**. 
    - U is zero, so we randomly choose an action.
    - We play a random game (turn blue player 1).
    - Blue player won.
    - So, we update V to **V=1**, **N=1**, **N<sub>tot</sub>=1**
    - Then, all the U values in all the branches need to be updated.
    - For the next iteration, we choose the branch with the **maximum U** (U=1.5 in this case).
    - This time, the node has **already been visited previously** and we would like to get some **more information** by exploring **deeper into the tree structure**.
    - So, instead of just playing another random game, we **consider all the possible next moves**.
        - Again, we choose the action with the maximum U, they're all zero in this case. So, we randomly choose an action and play a random game.
        - This time, the blue player won.
        - So we update V and N and U in the second level. 
    - Update the statistics on the previous node. This procedure is called **backpropagation**, where we go back up the tree to update all the variables.
        - First: Increase total number of visits (N=2)
        - Second: the expected outcome, V needs to be updated. We replace it by the average outcome of all the games that are played from this node keeping in mind that the outcome is from the perspective of the current player, orange in this case --> **V=0** (Because in the previous playout, the blue player won)
        - N total needs to be updated with **N<sub>tot</sub>=2** 
        - All the Us also needs to be updated.
    - Now, we can repeat this process again and again until a fixed number of total games are played.
    - Then to choose the next move, we pick the node with the highest visit count, 21 in this case.
    - This is the **move that we will choose** according to the **tree search**.

    ![image5]

- One advantage of expansion and propagation is that if we want to utilize **Monte Carlo tree search again after making a move**, we **don't need to start from scratch**. As we can **reuse previous results** by replacing the top node with the chosen node for the next move and we can continue our Monte Carlo tree search.
    ![image6]

    ![image7]


## AlphaZero 1: Guided Tree Search <a name="guided_tree_search"></a> 
A pure Monte Carlo search for Go would never be able to accumulate enough statistics to be reliable. So, we need to find a better way to improve Monte Carlo tree search.

![image8]
 
### Deep neural networks:
- We introduce an **expert policy &pi;<sub>&theta;</sub>**. Actor tells us the **probability of the next action** by an expert player,
- We introduce an **expert critic v<sub>&theta;</sub>**. Critic tells us what the **expected score** is from the perspective of the current player.

- The idea is that
    - using the **policy**, we can focus on **exploring actions** that an expert is likely to play.
    - using the **critic** will allow us to **estimate the outcome of a game** without running a simulation.

- Both the **policy and the critic come from a single neural network**, and thus, they share the **same weights &theta;** (see AlphaZero paper)
- Now, we can choose an action (using the policy) and optimize the policy
- To do this, we perform a Monte Carlo tree search, utilizing both the policy and the critic, so that the resulting moves are stronger than the policy alone, then we can use it to update the policy.

    ![image9]

### Example Tic-Tac-Toe:
- Given a state, we want to find out what the best action is.
- Just like the case of Monte-Carlo tree search, we consider all the possible actions, and assign three numbers, U, N, and V to each of them.
- The U function is a bit different from the ones from normal Monte Carlo tree search.
    - We have the same first term, V again, and it controls exploitation. We'll see that this V will largely come from the critic.
    - The second term is the exploration term that encourages visiting nodes with small visit counts. This is proportional to the policy. so that moves that are more likely to be played by an expert will be favored. Notice that we also introduced a hyperparameter C, that controls the level of exploration.

    ![image10]

- Let's see the search in action.
    - In the beginning, all the U's are zero.
    - So we randomly choose an action.
    - Then we can compute an estimated outcome using the critic **v<sub>&theta;</sub>**, and update all the values in that node.
    - The total number of visits is now **N<sub>tot</sub>=1**.
    - So, the exploration terms for all the other branches need to be updated **U=c&pi;<sub>&theta;</sub>**

    ![image11]

    - To iterate this process, we need to compare all the U's, and visit the branch with the largest U again.
    - The search may lead us to a different branch.
    - Again, we update the expected outcome using the critic, and the U function for the other branches.

    ![image12]

    - As this process is repeated, we may revisit the same branch again, what do we do then?
    - In order to get more information, we can expand this node into all the possible subsequent states, each with their own values U, N, and V, all initialized to zero.
    - Then we can choose an action with the maximum U, and compute the expected outcome from the critic.

    ![image13]

    - Afterward, we can go back to the original node and update the expected outcome.
    - The V can be computed as the average of the initial estimate through the critic, together with the results of exploring all the children nodes.
    - Notice that when computing the average for V, there's a **negative sign in front of the contributions from the children nodes**.
    - This is necessary, because by convention, V is an estimate from the perspective of the current player and going to a trial node, changes the player.

    ![image14]

    - After many iterations, we might reach a terminal state where the game ends.
    - Then, instead of using the critic for estimating the expected outcome, we simply use the actual outcome.
    - We can repeat this process for a fixed number of games and total.
    - Just like in Monte Carlo tree search, to choose the best action, we can simply pick the **action with the highest visit counts**,
    - Or if we want to encourage more exploration, we can choose an action stochastically with a probability proportional to the number of visits for each action, like the equation here:

    ![image15]


## AlphaZero 2: Self-Play Training <a name="self_playing"></a> 
- Now that we have an **improved Monte-Carlo Tree Search** guided by an **expert policy and critic**,
how do we update them?
    - Start with an empty board of Tic-Tac-Toe,
    - We perform **Monte-Carlo Tree Search** using the **current policy and critic**.
    - In the end we get a list of visit counts for each actions **N<sub>a</sub>**,
    - This list can be converted into a list of probabilities **p<sub>a</sub><sup>(t)</sup>** for each action at every time step.
    - After choosing the first action, we can perform Monte-Carlo Tree Search again.
    - Now, we don't have to start from scratch because this current state is likely to have a very high visit counts and many of the subsequent states should have been explored already.
    - So, we can iterate the Monte-Carlo Tree Search algorithm fairly efficiently.
    - Eventually, we repeat this process and arrive at the end game.
    - In this case, the final score is z equals **z=+1**.
    - The **final outcome** can be **compared to** the **expected outcome** computed at **each time step** using the **critic**.
    - We also computed the probability of performing an action through the Monte-Carlo Tree Search at each time step, and that can be compared to the expert policy as well.
    - The loss function **L<sub>&theta;</sub>** contains two terms:
        - The first term is the square of the difference between the **prediction from the critic** and the **actual outcome**.
        - The second term consists of the logarithm of the **policy** can be interpreted as an **entropy loss** between the **current policy** and the **probabilities computed through Monte-Carlo Tree Search**.
    - Minimizing this term, forces the policy to be closer to the results of Monte-Carlo Tree Search, and thus, strengthening the actions predicted by the policy.
    - Using the loss function, we can perform gradient descent to update both the policy and the critic.

    ![image16]

### AlphaZero Algorithm
- Below you can find a summary of the AlphaZero alkgorithm

    ![image17]

### How does learning happen intuitively?
- Starting from random critic and policy,the MCTS in AlphaZero should be not better than a standard Monte-Carlo Tree Search. How does it learn then?
Answer:
- In the beginning, the **critic is able to improve** because whenever we reach end game during the Tree Search,
the **end game outcome is propagated in the tree** and the critic will be able to predict this outcome better and better.
- After the **critic becomes less random**, the **policy will start to improve** as well.
- As training goes on, the AlphaZero agent will **first learn how to play the end game** very well.
- As the **end game improves**, the **mid game will improve** as well.
- Eventually, the algorithm will be able to anticipate long sequences of expert moves, leading to an expert level of gameplay

## TicTacToe using AlphaZero - code <a name="tic_tac_toe_code"></a> 
### Files in the repo
- ***alphazero-TicTacToe.ipynb***: Jupyter notebook as the main file 
- ***ConnectN.py***: Python file to implement a gameplay environment. Class containing game state, score the player.
- ***MCTS.py***: Python file to implement a Monte-Carlo Tree Search algorithm for AlphaZero. Inside the Node class there are functions to explore the tree, picking moves from the tree etc..
- ***Play.py***: Python file to implement an interactive environment for you to play the game (play against yourself, against an agent, or watch two agents playing against each other).

### Main File: alphazero-TicTacToe.ipynb
- Open Jupyer Notebook ```alphazero-TicTacToe.ipynb```
    ### Setup a game
    ```
    from ConnectN import ConnectN

    # 3x3 board, winning condition: three in a row
    game_setting = {'size':(3,3), 'N':3}
    game = ConnectN(**game_setting)

    game.move((0,1))
    print(game.state)    # 3x3 matrix
    print(game.player)   # alternating player 1 or -1
    print(game.score)    # game.score == None, when game is not over, +1 --> Player1 won, -1 --> Player2 won

    RESULTS:
    ------------
    [[ 0.  1.  0.]
    [ 0.  0.  0.]
    [ 0.  0.  0.]]
    -1
    None
    ```
    ### Some examople actions (moves)
    ```
    # player -1 move
    game.move((0,0))
    # player +1 move
    game.move((1,1))
    # player -1 move
    game.move((1,0))
    # player +1 move
    game.move((2,1))

    print(game.state)
    print(game.player)
    print(game.score)

    RESULTS:
    ------------
    [[-1.  1.  0.]
    [-1.  1.  0.]
    [ 0.  1.  0.]]
    1
    1
    ```
    ### Create Neural network for Policy and Critic 
    ```
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from math import *
    import numpy as np
    import random


    class Policy(nn.Module):
        """ Create a neural network for lPolicy and Critic
        """

        def __init__(self):
            """ Init function of class Policy to intialize important neural network parameters
                
                INPUTS:
                ------------
                    None
                    
                OUTPUTS:
                ------------
                    No direct
            """
            
            super(Policy, self).__init__()
            
            # first layers of the neural network
            self.conv = nn.Conv2d(1, 16, kernel_size=2, stride=1, bias=False)
            self.size = 2*2*16
            self.fc = nn.Linear(self.size,32)

            # layers for the policy
            self.fc_action1 = nn.Linear(32, 16)
            self.fc_action2 = nn.Linear(16, 9)
            
            # layers for the critic
            self.fc_value1 = nn.Linear(32, 8)
            self.fc_value2 = nn.Linear(8, 1)
            self.tanh_value = nn.Tanh()
            
            
        def forward(self, x):
            """ Forward pass of neural network
                
                INPUTS:
                ------------
                    x - (torch tensor) state tensor e.g.    tensor([[[[ 0.,  0.,  0.],
                                                                    [ 0.,  0.,  0.],
                                                                    [ 0.,  0.,  0.]]]])
                OUTPUTS:
                ------------
                    out_policy_critic - (tuple of torch tensors) e.g.  (tensor([[ 0.1217,  0.1371,  0.1514],
                                                                                [ 0.1118,  0.1013,  0.0997],
                                                                                [ 0.1251,  0.0000,  0.1518]]), tensor([[-0.3396]]))

            """
            
            # first layers of the neural network
            y = F.relu(self.conv(x))
            y = y.view(-1, self.size)
            y = F.relu(self.fc(y))
            
            # the action head (policy)
            a = F.relu(self.fc_action1(y))
            a = self.fc_action2(a)
            # availability of moves, this matrix gives 0 when move not available, and 1 if it is available
            avail = (torch.abs(x.squeeze())!=1).type(torch.FloatTensor)
            avail = avail.view(-1, 9)
            
            # locations where actions are not possible, we set the prob to zero
            # If state is already occupied, this state cannot be occupied a second time
            maxa = torch.max(a)
            # subtract off max for numerical stability (avoids blowing up at infinity)
            exp = avail*torch.exp(a-maxa)
            prob = exp/torch.sum(exp)
            
            # the value head (critic)
            value = F.relu(self.fc_value1(y))
            value = self.tanh_value(self.fc_value2(value))
            out_policy_critic = prob.view(3,3), value
        
            return out_policy_critic
            

    policy = Policy()
    ```
    ### Define a player that uses MCTS and the expert policy + critic to play a game
    ```
    import MCTS

    from copy import copy
    import random

    def Policy_Player_MCTS(game):
        """ Initialize a player that can use MCTS to play a game
        
            INPUTS:
            ------------
                game - (instance of ConnectN.ConnectN)
            
            OUTPUTS:
            ------------
                mytreenext.game.last_move
        """
    
        # Make a copy and initialize a Monte Carlo tree search class
        mytree = MCTS.Node(copy(game))
        for _ in range(50):
            # Explore the tree by supplying a policy
            # All Us will be computed, take the branch with the maximal U, search, expand, backpropagate, increase N  
            mytree.explore(policy)
    
        # Tell the tree to choose the next move choose action with largest visit count
        mytreenext, (v, nn_v, p, nn_p) = mytree.next(temperature=0.1)
        
        # return a move
        return mytreenext.game.last_move

    def Random_Player(game):
        return random.choice(game.available_moves())    
    ```
    ### Create a new game and instantiate a MCTS Policy_Player
    ```
    game = ConnectN(**game_setting)
    print(game.state)
    Policy_Player_MCTS(game)

    RESULTS:
    -------------
    state:
    [[ 0.  0.  0.]
     [ 0.  0.  0.]
     [ 0.  0.  0.]]

    action:
    (0, 2)
    ```
    ### Train the agent
    ```
    # initialize our alphazero agent and optimizer
    import torch.optim as optim

    game=ConnectN(**game_setting)
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=.01, weight_decay=1.e-4)
    ```
    ```
    # train our agent

    from collections import deque
    import MCTS

    episodes = 400
    outcomes = []
    losses = []

    !pip install progressbar
    import progressbar as pb
    widget = ['training loop: ', pb.Percentage(), ' ', 
            pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()

    for e in range(episodes):
        # for each episode initialize a top node for the Monte Carlo tree search
        mytree = MCTS.Node(ConnectN(**game_setting))
        vterm = []
        logterm = []
        
        # as long as the game is not over, explore up to 50 steps
        while mytree.outcome is None:
            for _ in range(50):
                mytree.explore(policy)

            # keep track of the player 
            current_player = mytree.game.player
            
            # increment to the next tree
            # next() --> the tree was to choose an action based on the N 
            # Outputs: v, nn_v, p, nn_p --> used to update the policy
            # v --> expected outcome (computed from tree search, not useful here), 
            # nn_v --> the critic value evaluating the current board 
            # p --> action porobability list of taking each action computed by MCTS
            # nn_p --> same as p butcoming straight from the policy
            mytree, (v, nn_v, p, nn_p) = mytree.next()        
            mytree.detach_mother()
            

            # solution: compute a loss function by comparing p with nn_p
            # compute prob* log pi 
            loglist = torch.log(nn_p)*p
            
            # constant term to make sure if policy result = MCTS result, loss = 0
            constant = torch.where(p>0, p*torch.log(p),torch.tensor(0.))
            logterm.append(-torch.sum(loglist-constant))
            
            # add the critic value, factor 'current_player' is needed to calculate the crtic value from the perspective of the current player
            vterm.append(nn_v*current_player)
            
            
        # we compute the "policy_loss" for computing gradient
        outcome = mytree.outcome
        outcomes.append(outcome)
        
    
        # loss: two terms --> critic (prediction from the critic - actual outcome)^2 + cross entropy loss (between current policy and action probability)
        loss = torch.sum( (torch.stack(vterm)-outcome)**2 + torch.stack(logterm) )
        optimizer.zero_grad()
        
        # backpropagation
        loss.backward()
        losses.append(float(loss))
        
        # Update policy
        optimizer.step()
        
        if (e+1)%50==0:
            print("game: ",e+1, ", mean loss: {:3.2f}".format(np.mean(losses[-20:])),
                ", recent outcomes: ", outcomes[-10:])
        del loss
        
        timer.update(e+1)
        
        
    timer.finish()
    ```
    ### Plot the loss 
    ```
    # plot your losses

    import matplotlib.pyplot as plt

    % matplotlib notebook
    plt.plot(losses)
    plt.show()
    ```
    ![image18]

    ### Play a game against the alphazero agent 
    ```
    % matplotlib notebook

    # as first player
    gameplay=Play(ConnectN(**game_setting), 
                player1=None, 
                player2=Policy_Player_MCTS)
    ```

### Implementation of the game class: ConnectN.py
- Open Python file ```ConnectN.py```
    ### ConnectN class:
    ```
    class ConnectN:
        """ Implement a game
        """
        def __init__(self, size, N, pie_rule=False):
            """ Init of ConnectN class, initialize size of the game board, the winning conditions, etc.

                INPUTS: 
                ------------
                    size - (tuple) here (3,3)
                    N - (int) here 3
                    pie_rule=False
                    
                OUTPUTS:
                ------------
                    No direct  
            """
            self.size = size
            self.w, self.h = size
            self.N = N

            # make sure game is well defined
            if self.w<0 or self.h<0 or self.N<2 or \
                (self.N > self.w and self.N > self.h):
                raise ValueError('Game cannot initialize with a {0:d}x{1:d} grid, and winning condition {2:d} in a row'.format(self.w, self.h, self.N))

            self.score = None
            self.state=np.zeros(size, dtype=np.float)
            self.player=1
            self.last_move=None
            self.n_moves=0
            self.pie_rule=pie_rule
            self.switched_side=False

        # fast deepcopy
        def __copy__(self):
            
            """ Function to copy the class instance quickly. Needed for Monte-Carlo tree search to explore
                
                INPUTS:
                ------------
                    None
                
                OUTPUTS:
                ------------
                    new_game - (copied instance of ConnectN)
            
            """
            cls = self.__class__
            new_game = cls.__new__(cls)
            new_game.__dict__.update(self.__dict__)

            new_game.N = self.N
            new_game.pie_rule = self.pie_rule
            new_game.state = self.state.copy()
            new_game.switched_side = self.switched_side
            new_game.n_moves = self.n_moves
            new_game.last_move = self.last_move
            new_game.player = self.player
            new_game.score = self.score
            return new_game

        # check victory condition
        # fast version
        def get_score(self):
            
            """ Compute scores from the last move. 
                It takes the last move and computes horizontal, vertical and diagonal lines around the last move.
                
                INPUTS:
                -------------
                    None 
                
                OUTPUTS:
                -------------
                    actual score, i.e. 0, -1 or +1
            """

            # game cannot end beca
            if self.n_moves<2*self.N-1:
                return None

            i,j = self.last_move
            hor, ver, diag_right, diag_left = get_lines(self.state, (i,j))

            # loop over each possibility
            for line in [ver, hor, diag_right, diag_left]:
                if in_a_row(line, self.N, self.player):
                    return self.player
                    
            # no more moves
            if np.all(self.state!=0):
                return 0

            return None

        # for rendering
        # output a list of location for the winning line
        def get_winning_loc(self):


            """ For rendering, i.e. displaying the graphics. 
                When game has been finished, it outputs a list of locations for the winning line
            
                INPUTS:
                ------------
                    None 
                    
                OUTPUTS:
                ------------
                    indices (list) of locations for the winning line
            """

            if self.n_moves<2*self.N-1:
                return []

            
            loc = self.last_move
            hor, ver, diag_right, diag_left = get_lines(self.state, loc)
            ind = np.indices(self.state.shape)
            ind = np.moveaxis(ind, 0, -1)
            hor_ind, ver_ind, diag_right_ind, diag_left_ind = get_lines(ind, loc)
            # loop over each possibility

            pieces = [hor, ver, diag_right, diag_left]
            indices = [hor_ind, ver_ind, diag_right_ind, diag_left_ind]

            #winning_loc = np.full(self.state.shape, False, dtype=bool)

            for line, index in zip(pieces, indices):
                starts, ends, runs = get_runs(line, self.player)

                # get the start and end location
                winning = (runs >= self.N)
                print(winning)
                if not np.any(winning):
                    continue
            
                starts_ind = starts[winning][0]
                ends_ind = ends[winning][0]
                indices = index[starts_ind:ends_ind]
                #winning_loc[indices[:,0], indices[:,1]] = True
                return indices
            
            return []


        def move(self, loc):

            """ Advance the game by one step
            
                INPUTS:
                ------------
                    loc - (tuple) of locations
                
                OUTPUTS:
                ------------
            """
            i,j=loc
            success = False
            if self.w>i>=0 and self.h>j>=0:
                if self.state[i,j]==0:

                    # make a move
                    self.state[i,j]=self.player

                    # if pie rule is enabled
                    if self.pie_rule:
                            if self.n_moves==1:
                                self.state[tuple(self.last_move)]=-self.player
                                self.switched_side=False
                    
                            elif self.n_moves==0:
                                # pie rule, make first move 0.5
                                # this is to let the neural net know
                                self.state[i,j]=self.player/2.0
                                self.switched_side=False
                            
                    success = True

                # switching side
                elif self.pie_rule and self.state[i,j] == -self.player/2.0:

                    # make a move
                    self.state[i,j]=self.player
                    self.switched_side=True

                    success = True

                            
                

            if success:
                self.n_moves += 1
                self.last_move = tuple((i,j))
                self.score = self.get_score()

                # if game is not over, switch player
                if self.score is None:
                    self.player *= -1
                
                return True

            return False


        def available_moves(self):
            
            """ Helper function: which states are available
                
                INPUTS:
                ------------
                    None 
                    
                OUTPUTS:
                ------------
                    indices - (numpy array) of available states
            """
            indices = np.moveaxis(np.indices(self.state.shape), 0, -1)
            return indices[np.abs(self.state) != 1]

        def available_mask(self):
            return (np.abs(self.state) != 1).astype(np.uint8)

    ```
### Most important file: MCTS.py
- Open Python file ```MCTS.py```
    ```
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.animation as animation
    from copy import copy
    from math import *
    import random

    c=1.0

    # transformations
    # board and game play should not change when we rotate or flip the board (square 3x3 board).
    # every time we put the game state to the neural network, we randomly choose one transformation together with one inverse 
    # transformation and apply it to that state. This speeds up training.
    t0= lambda x: x
    t1= lambda x: x[:,::-1].copy()
    t2= lambda x: x[::-1,:].copy()
    t3= lambda x: x[::-1,::-1].copy()
    t4= lambda x: x.T
    t5= lambda x: x[:,::-1].T.copy()
    t6= lambda x: x[::-1,:].T.copy()
    t7= lambda x: x[::-1,::-1].T.copy()

    tlist=[t0, t1,t2,t3,t4,t5,t6,t7]
    tlist_half=[t0,t1,t2,t3]

    def flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    # inversion --> flipping player, states, rewards etc.
    t0inv= lambda x: x
    t1inv= lambda x: flip(x,1)
    t2inv= lambda x: flip(x,0)
    t3inv= lambda x: flip(flip(x,0),1)
    t4inv= lambda x: x.t()
    t5inv= lambda x: flip(x,0).t()
    t6inv= lambda x: flip(x,1).t()
    t7inv= lambda x: flip(flip(x,0),1).t()

    tinvlist = [t0inv, t1inv, t2inv, t3inv, t4inv, t5inv, t6inv, t7inv]
    tinvlist_half=[t0inv, t1inv, t2inv, t3inv]

    transformation_list = list(zip(tlist, tinvlist))
    transformation_list_half = list(zip(tlist_half, tinvlist_half))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    device ='cpu'

    def process_policy(policy, game):
        """ Function to initiate a random sampling from transformation_list or transformation_list_half 
        
            INPUTS:
            ------------
                policy - (instance of class policy) actual policy
                game - (instance of ConnectN class) actual game
                
            OUTPUTS:
            ------------
                game.available_moves()
                tinv(prob)[mask].view(-1)
                v.squeeze().squeeze()
        """

        # for square board, add rotations as well
        if game.size[0]==game.size[1]:
            t, tinv = random.choice(transformation_list)

        # otherwise only add reflections
        else:
            t, tinv = random.choice(transformation_list_half)
        
        frame=torch.tensor(t(game.state*game.player), dtype=torch.float, device=device)
        input=frame.unsqueeze(0).unsqueeze(0)
        prob, v = policy(input)
        mask = torch.tensor(game.available_mask())
        
        # we add a negative sign because when deciding next move,
        # the current player is the previous player making the move
        return game.available_moves(), tinv(prob)[mask].view(-1), v.squeeze().squeeze()

    class Node:
        """ Implementation of the Monte-Carlo tree
        """
        def __init__(self, game, mother=None, prob=torch.tensor(0., dtype=torch.float)):
            """ Init function of Node class
            
                INPUTS:
                ------------
                    game - (instance of ConnectN class) actual game
                    mother - (None) not needed here
                    prob - (torch tensor) initialized with 0. 
                
                OUTPUTS:
                ------------
            """
            self.game = game
            
            # child nodes
            self.child = {}
            # numbers for determining which actions to take next
            self.U = 0

            # V from neural net output
            # it's a torch.tensor object
            # has require_grad enabled
            self.prob = prob
            # the predicted expectation from neural net
            self.nn_v = torch.tensor(0., dtype=torch.float)
            
            # visit count
            self.N = 0

            # expected V from MCTS
            self.V = 0

            # keeps track of the guaranteed outcome
            # initialized to None
            # this is for speeding the tree-search up
            # but stopping exploration when the outcome is certain
            # and there is a known perfect play
            self.outcome = self.game.score


            # if game is won/loss/draw
            if self.game.score is not None:
                self.V = self.game.score*self.game.player
                
                # all winning moves have U values equal to plus infinity
                # If there is a winning move we do not need to explore all the other non-winning moves
                self.U = 0 if self.game.score is 0 else self.V*float('inf')

            # link to previous node
            self.mother = mother

        def create_child(self, actions, probs):
            """ Create a dictionary of children
            
                INPUTS:
                ------------
                    actions (list)
                    probs - (list)
            """
            games = [ copy(self.game) for a in actions ]

            for action, game in zip(actions, games):
                game.move(action)

            child = { tuple(a):Node(g, self, p) for a,g,p in zip(actions, games, probs) }
            self.child = child
            
        def explore(self, policy):
            """ Explore the tree 
                Start from the top node
                Explore the childdren
                Pick children with maximum U
                If there are multiple nodes with maximum U --> randomly choose an action
                Increase the visit counts
                Update U and back-prop
                
                INPUTS:
                ------------
                    policy - (torch tensor) output from Policy class (policy + critic)
                
                OUTPUTS:
                ------------
                    No direct
            """

            if self.game.score is not None:
                raise ValueError("game has ended with score {0:d}".format(self.game.score))

            # Start from the top node
            current = self

            
            # explore children of the node
            # to speed things up 
            while current.child and current.outcome is None:

                child = current.child
                
                # Pick children with maximum U
                max_U = max(c.U for c in child.values())
                #print("current max_U ", max_U) 
                actions = [ a for a,c in child.items() if c.U == max_U ]
                if len(actions) == 0:
                    print("error zero length ", max_U)
                    print(current.game.state)
                
                # If there are multiple nodes with maximum U --> randomly choose an action
                action = random.choice(actions)            

                if max_U == -float("inf"):
                    current.U = float("inf")
                    current.V = 1.0
                    break
                
                elif max_U == float("inf"):
                    current.U = -float("inf")
                    current.V = -1.0
                    break
                    
                current = child[action]
            
            # if node hasn't been expanded
            if not current.child and current.outcome is None:
                # policy outputs results from the perspective of the next player
                # thus extra - sign is needed
                next_actions, probs, v = process_policy(policy, current.game)
                current.nn_v = -v
                current.create_child(next_actions, probs)
                current.V = -float(v)

            # Increase the visit counts
            current.N += 1

            # now update U and back-prop
            while current.mother:
                mother = current.mother
                mother.N += 1
                # beteen mother and child, the player is switched, extra - sign
                mother.V += (-current.V - mother.V)/mother.N

                #update U for all sibling nodes
                for sibling in mother.child.values():
                    if sibling.U is not float("inf") and sibling.U is not -float("inf"):
                        sibling.U = sibling.V + c*float(sibling.prob)* sqrt(mother.N)/(1+sibling.N)

                current = current.mother


                
        def next(self, temperature=1.0):
            """ Chooses the next action for the tree
                Find the maximum U
                If maximum U is infinity --> this is a winning move (speeds up training by telling the agent to always choose winning moves)
                If max U is not inf --> look for the max visit count --> take thsi action
                # Increment to the next step --> Random choice based on their probabilities
            
                INPUTS:
                ------------
                    temperature - (float) hyperparameter needed for action probability 
                    
                OUTPUTS:
                ------------
                    nextstate - (list with one element)
                    (-self.V, -self.nn_v, prob, nn_prob) - (tuple) needed to update the policy 
            """

            if self.game.score is not None:
                raise ValueError('game has ended with score {0:d}'.format(self.game.score))

            if not self.child:
                print(self.game.state)
                raise ValueError('no children found and game hasn\'t ended')
            
            child=self.child

            # Find the maximum U
            # if there are winning moves, just output those
            max_U = max(c.U for c in child.values())

            if max_U == float("inf"):
                prob = torch.tensor([ 1.0 if c.U == float("inf") else 0 for c in child.values()], device=device)
                
            else:
                # divide things by maxN for numerical stability
                maxN = max(node.N for node in child.values())+1
                prob = torch.tensor([ (node.N/maxN)**(1/temperature) for node in child.values() ], device=device)

            # normalize the probability
            if torch.sum(prob) > 0:
                prob /= torch.sum(prob)
                
            # if sum is zero, just make things random
            else:
                prob = torch.tensor(1.0/len(child), device=device).repeat(len(child))
            
            
            nn_prob = torch.stack([ node.prob for node in child.values() ]).to(device)
            
            # Increment to the next step --> Random choice based on their probabilities
            nextstate = random.choices(list(child.values()), weights=prob)[0]
            
            # V was for the previous player making a move
            # to convert to the current player we add - sign
            return nextstate, (-self.V, -self.nn_v, prob, nn_prob)

        def detach_mother(self):
            del self.mother
            self.mother = None
    ```

### The Play class: Do the rendering
- Open Python file ```Play.py```
    ### Most important function is play() --> see below
    ```
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    import numpy as np
    import time

    from copy import copy

    class Play:
        """ Class needed for rendering
        """
        
        def __init__(self, game, player1=None, player2=None, name='game'):
            self.original_game=game
            self.game=copy(game)
            self.player1=player1
            self.player2=player2
            self.player=self.game.player
            self.end=False
            self.play()

        def reset(self):
            self.game=copy(self.original_game)
            self.click_cid=None
            self.end=False
            
        def play(self, name='Game'):
            """ Function set starts rendering. It checks for None.
                if one player is None the click() function is needed
                if one player is not None --> FuncAnimation is triggered (draws each move as the move is played)
            """
            
            self.reset()
            
            if self.game.w * self.game.h <25:
                figsize=(self.game.w/1.6, self.game.h/1.6)

            else:
                figsize=(self.game.w/2.1, self.game.h/2.1)

            
            self.fig=plt.figure(name, figsize=figsize)
            if self.game.w * self.game.h <25:
                self.fig.subplots_adjust(.2,.2,1,1)
            else:
                self.fig.subplots_adjust(.1,.1,1,1)
                
            self.fig.show()
            w,h=self.game.size
            self.ax=self.fig.gca()
            self.ax.grid()
            # remove hovering coordinate tooltips
            self.ax.format_coord = lambda x, y: ''
            self.ax.set_xlim([-.5,w-.5])
            self.ax.set_ylim([-.5,h-.5])
            self.ax.set_xticks(np.arange(0, w, 1))
            self.ax.set_yticks(np.arange(0, h, 1))
            self.ax.set_aspect('equal')
        
            for loc in ['top', 'right', 'bottom', 'left']:
                self.ax.spines[loc].set_visible(False)


            # fully AI game
            if self.player1 is not None and self.player2 is not None:
                self.anim = FuncAnimation(self.fig, self.draw_move, frames=self.move_generator, interval=500, repeat=False)
                return
            
            # at least one human
            if self.player1 is not None:
                # first move from AI first
                succeed = False
                while not succeed:
                    loc = self.player1(self.game)
                    succeed = self.game.move(loc)

                self.draw_move(loc)
                
            self.click_cid=self.fig.canvas.mpl_connect('button_press_event', self.click)

                
        def move_generator(self):
            score = None
            # game not concluded yet
            while score is None:
                self.player = self.game.player
                if self.game.player == 1:
                    loc = self.player1(self.game)
                else:
                    loc = self.player2(self.game)
                    
                success = self.game.move(loc)

                # see if game is done
                if success:
                    score=self.game.score
                    yield loc
                    
            
        def draw_move(self, move=None):
            if self.end:
                return
            
            i,j=self.game.last_move if move is None else move
            c='salmon' if self.player==1 else 'lightskyblue'
            self.ax.scatter(i,j,s=500,marker='o',zorder=3, c=c)
            score = self.game.score
            self.draw_winner(score)
            self.fig.canvas.draw()


        def draw_winner(self, score):
            if score is None:
                return
            
            if score == -1 or score == 1:
                locs = self.game.get_winning_loc()
                c='darkred' if score==1 else 'darkblue'
                self.ax.scatter(locs[:,0],locs[:,1], s=300, marker='*',c=c,zorder=4)

            # try to disconnect if game is over
            if hasattr(self, 'click_cid'):
                self.fig.canvas.mpl_disconnect(self.click_cid)

            self.end=True
            
        
        def click(self,event):
            
            loc=(int(round(event.xdata)), int(round(event.ydata)))
            self.player = self.game.player
            succeed=self.game.move(loc)

            if succeed:
                self.draw_move()

            else:
                return
            
            if self.player1 is not None or self.player2 is not None:

                succeed = False
                self.player = self.game.player
                while not succeed:
                    if self.game.player == 1:
                        loc = self.player1(self.game)
                    else:
                        loc = self.player2(self.game)
                    succeed = self.game.move(loc)
                
                self.draw_move()
    ```

## Advanced TicTacToe with AlphaZero - Walkthrough <a name="tic_tac_toe_adv"></a> 
- Check out the [Code-Walkthrough-Video](https://www.youtube.com/watch?time_continue=430&v=MOIk_BbCjRw&feature=emb_logo)
- Open Jupyter Notebook ```alphazero-TicTacToe-advanced.ipynb```


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

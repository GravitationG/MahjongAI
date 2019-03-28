# MahjongAI
In this project, we have implemented several reinforce learning methods to solve the Mahjong game. Our environment is based on the project of https://github.com/clarkwkw/mahjong-ai. We used its Game.py , Player.py, HKRules.py, Tile.py, MoveGenerator.py to build the pipeline of this complicated game. 

## Monte Carlo Tree Search method 
In the folder of MCTS_method, we implemented the Monte Carlo Tree Search method for the agent to play the Mahjong game. 
The MonteCarloTree_move.py is the implementation of interface MoveGenerator in MoveGenerator.py. 
You can download and run the test.ipynb to check out the game test with RandomGenerator and our implementation MCTS_Generator. 

## Deep Q Learning method
In the folder of DeepQLearning, we inplemented the Deep Q-Learing algorithme. To train the model, please run the scricpt Training.py. Network.py is the deep network we utilize; Generator.py is the implementation of MoveGenerator. 
Dependency:
Tensorflow

## Policy Gradient and Actor Critic method 
In the folder of PolicyGradient_ActorCritic, we have implemented the method of Policy Gradient and Actor Critic.
To run the code, please open the jupyter notebook respectively.
Dependency:
Tensorflow

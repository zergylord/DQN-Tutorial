Welcome to the Q-learning CogSci Tutorial!

Code implements basic q-learner with non-linear function approximation (neural network)

Some challenges:
-Implement a replay buffer: Instead of updating the network based on the current state,action,reward,next-state tuple, store 
these in a big numpy array, and update based on randomly sampled minibatches of these tuples
-Implement a target network: Setup a 2nd computational graph with the same structure as the main structure, and use TensorFlow's
'assign' command to copy the weights of the first network every 10,000 steps. Experiment with different schedules, or try having the
2nd network slowly follow the 1st! Use this target network for the target values (i.e. r+gamma*Q(sPrime,aMax))

pretty code of a completed implementation of DQN (not made by me): https://github.com/devsisters/DQN-tensorflow

Notice the encapulation of functionality across functions and files, and how everything is documented. You should do this too.

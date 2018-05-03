# PolicyGradient
## Implementation of Policy Gradient Learning algorithm on a 2D test-bed.

Requirements : gym,numpy,argparse,matplotlib
	
usage :
	
	usage: rollout.py [-h] [--alpha ALPHA] [--gamma GAMMA] [--episodes EPISODES]
                  [--env ENV] [--render] --batchSize BATCHSIZE

	ALPHA : learning rate [Default:0.5]
	GAMMA : Discount [Default:0.9]
	--render : to render the game for last 10% of the episodes
	ALGO : Q/SARSA/SARSAlam
	ENV : [chakra/visham]
	
Example : To run a game with default settings on grid-A, to render the enviroment and show learnt policy at the end
	
	python rollout.py --episodes 2000 --render --env chakra

Reference : [Reinforcement Learning: An Introduction Book by Andrew Barto and Richard S. Sutton](http://incompleteideas.net/book/bookdraft2018mar21.pdf)

Credits : The TAs of CS6700 Spring 2018 IITM for the Gym-Environment.	
		

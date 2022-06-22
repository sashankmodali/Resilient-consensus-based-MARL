# Multi Agent Reinforcement Learning in a grid world (adopted from [Mfigura, RPBCAC](https://github.com/mfigura/Resilient-consensus-based-MARL))

The repository contains folders whose description is provided below:

1) agents - contains resilient and adversarial agents
2) environments - contains a grid world environment for the cooperative navigation task
3) simulation_results - contains plots that show training performance
4) training - contains functions for training agents

To train agents, execute main.py.

## Multi-agent grid world: cooperative navigation
We train five agents in a grid-world environment. Their original goal is to approach their desired position without colliding with other agents in the network.
We design a grid world of dimension (5 x 5) and consider a reward function that penalizes the agents for distance from the target and colliding with other agents.

<img src="https://github.com/sashankmodali/grid-world-MARL/blob/main/simulation_results/illustrations/cooperative_navigation.jpg" width="440" align="left">
<img src="https://github.com/sashankmodali/grid-world-MARL/blob/main/simulation_results/illustrations/com_graph.jpg" width="300" >

## References

<a id="1">[1]</a>
M. Figura, Y. Lin, J. Liu, V. Gupta,
Resilient Consensus-based Multi-agent Reinforcement Learning with Function Approximation.
arXiv preprint arXiv:2111.06776, 2021.

<a id="2">[2]</a> 
M. Figura, K. C. Kosaraju and V. Gupta,
Adversarial attacks in consensus-based multi-agent reinforcement learning,
2021 American Control Conference (ACC), 2021.

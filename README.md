# Capture Uncertainty in Dating
This project aims to model the two-sided matching between Romeo and Juliet, with individuals categorized as either type H or type L. The probability of a successful match between a Romeo and Juliet depends solely on their respective types.
To learn the probability of each individual's type, a Bayesian approach is used based on multiple attempts at pairing individuals selected by the algorithm from the other side. 

## Algorithms
### Greedy
The Greedy algorithm selects the most promising pairing method and generates pairs based on temporary classifications.
### Random
The Random algorithm generates dating pairs randomly and updates the posterior of all agents after each round. In the final round, the algorithm generates Greedy pairs using the posterior probabilities.

### Epsilon-greedy
The epsilon-greedy algorithm has a probability epsilon of choosing a non-greedy way to generate pairs. For example, if the Greedy choice is to put a type H Romeo and a type H Juliet together, when the algorithm decides to deviate, it may put a type H Romeo and a type L Juliet together instead.

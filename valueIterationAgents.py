# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        states = self.mdp.getStates()
        for i in range(self.iterations):
            values = self.values.copy()
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                maxVal = -float('inf')
                for a in actions:
                    val = self.computeQValueFromValues(s, a)
                    if val > maxVal:
                        maxVal = val
                if maxVal == -float('inf'):
                    values[s] = 0
                else:
                    values[s] = maxVal
            self.values = values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        tAndS = self.mdp.getTransitionStatesAndProbs(state, action)
        val = 0
        for sprime, t in tAndS:
            r = self.mdp.getReward(state, action, sprime)
            val += t * (r + (self.discount * self.values[sprime]))
        return val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        maxAc = None
        maxQ = -float('inf')
        for a in actions:
            q_value = self.computeQValueFromValues(state, a)
            if q_value > maxQ:
                maxQ = q_value
                maxAc = a
        return maxAc

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        index = 0
        for i in range(self.iterations):
            values = self.values.copy()
            s = self.mdp.getStates()[index]
            index += 1
            if index == len(self.mdp.getStates()):
                index = 0
            if not(self.mdp.isTerminal(s)):
                actions = self.mdp.getPossibleActions(s)
                maxVal = -float('inf')
                for a in actions:
                    val = self.computeQValueFromValues(s, a)
                    if val > maxVal:
                        maxVal = val
                if maxVal == -float('inf'):
                    values[s] = 0
                else:
                    values[s] = maxVal
                self.values[s] = values[s]


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        queue = util.PriorityQueue()
        states = self.mdp.getStates()
        predecessors = dict()
        for s in states:
            predecessors[s] = set()
        for s in states:
            actions = self.mdp.getPossibleActions(s)
            for a in actions:
                tAndS = self.mdp.getTransitionStatesAndProbs(s, a)
                for sprime, t in tAndS:
                    if t > 0:
                        predecessors[sprime].add(s)
            if not (self.mdp.isTerminal(s)):
                curVal = self.values[s]
                maxAc = self.computeActionFromValues(s)
                maxQ = self.computeQValueFromValues(s, maxAc)
                diff = abs(curVal - maxQ)
                queue.update(s, -diff)
        for i in range(self.iterations):
            if queue.isEmpty():
                return
            s = queue.pop()
            if not (self.mdp.isTerminal(s)):
                maxAc = self.computeActionFromValues(s)
                maxQ = self.computeQValueFromValues(s, maxAc)
                self.values[s] = maxQ
            for p in predecessors[s]:
                curVal = self.values[p]
                maxAc = self.computeActionFromValues(p)
                maxQ = self.computeQValueFromValues(p, maxAc)
                diff = abs(curVal - maxQ)
                if diff > self.theta:
                    queue.update(p, -diff)

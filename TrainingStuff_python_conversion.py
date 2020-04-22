#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Necessary Imports
from pacman import GhostRules, PacmanRules, ClassicGameRules, GameState
from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from game import Agent
from util import nearestPoint
from util import manhattanDistance
import util, layout
import sys, types, time, random, os

from collections import deque

from TrainingStuffs import *
import numpy as np
from Constants import *
import tensorflow as tf
from tensorflow import keras
import random

print(tf.__version__)
print("Num GPUs Available: " + str(len(tf.config.experimental.list_physical_devices('GPU'))))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(str(len(gpus)) + " Physical GPUs, " + str(len(logical_gpus)) + " Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# In[2]:


# Pacman Agent, or custom defined if needed
# @ Sergey, key thing to note, you should 
# always check if a resulting action is legal
# Else just have pacman stay in place
# This basically allows the network to just keep 
# Pacman in a cubby for a bit
class SmartAgent(Agent):
    
    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature
        self.last_input = None
        
    def init_training(self, model):
        self.is_train = True
        self.target_model = model
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
    
    def getAction(self, state):
        network_input = convert_state_to_input(state, self.last_input)
        self.last_input = network_input
        
        predictions = self.model.predict(np.array([network_input]))[0]
        probs = tf.nn.softmax(predictions).numpy()
        
        new_probs = np.zeros(probs.shape)
        prob_sum = 0.0
        power = 1.0 / self.temperature
        for i in range(len(probs)):
            p = probs[i] ** power
            prob_sum += p
            new_probs[i] = p
        probs = new_probs / prob_sum
        
        move = select_from_distribution(probs)
        
        action = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST][move]
        if action in state.getLegalPacmanActions():
            return action
        else:
            legals = list(state.getLegalPacmanActions())
            legals.remove(Directions.STOP)
            return random.choice(legals)
        
    def update_memory(self, action, next_state, reward, done):
        if self.is_train:
            self.replay_memory.append((self.last_input, action, convert_state_to_input(next_state, self.last_input), reward, done))
    
    def train(self, is_terminal_state):
        if not self.is_train:
            return
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        
        new_current_states = np.array([transition[2] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []
        
        actionIndexMap = {
            Directions.NORTH: 0, 
            Directions.EAST: 1, 
            Directions.SOUTH: 2, 
            Directions.WEST: 3
        }

        # Now we need to enumerate our batches
        for index, (current_state, action, new_current_state, reward, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
#             if action != Directions.STOP:
            current_qs[actionIndexMap[action]] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if is_terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


# In[3]:


class MyGhostAgent( Agent ):
    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        action = Directions.EAST # Replace This with a call to the model
        if action in state.getLegalActions( self.index ):
            return action
        else:
            return random.choice( state.getLegalActions( self.index ))


# In[4]:


#define the ghost agent ___otherwise import it here. 
class SmartGhost( MyGhostAgent ):
    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        action = Directions.EAST # Replace This with a call to the model
        if action in state.getLegalActions( self.index ):
            return action
        else:
            return random.choice( state.getLegalActions( self.index ))


# In[5]:


args = dict()
args['layout'] = layout.getLayout("originalClassic.lay")
if args['layout'] == None: raise Exception("The layout " + options.layout + " cannot be found")

def create_model_sequential_api():
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=16,
                            kernel_size=3,
                            strides=(1, 1),
                            data_format="channels_first",
                            activation="relu",
                            input_shape=(TIMESTEP_PLANES*INPUT_TIMESTEPS, PLANE_WIDTH, PLANE_HEIGHT)),
        keras.layers.Flatten(data_format="channels_first"),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(4, activation='tanh')
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(lr=LEARNING_RATE, decay=DECAY, momentum=MOMENTUM),
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )
    return model

smart_agent_model = SmartAgent(None, 1)
args['pacman'] = smart_agent_model

ghosts = [SmartGhost, MyGhostAgent,MyGhostAgent,MyGhostAgent]
args['ghosts'] = [ghosts[i](i+1) for i in range(len(ghosts))]

args['numTraining'] = 0
args['numGames'] = 100
args['record'] = True
args['catchExceptions'] = False
args['timeout'] = 30


# In[6]:


#define a "run" function for the iterations
import textDisplay
def runGames( layout, pacman, ghosts, numGames, record, numTraining = 0, catchExceptions=False, timeout=30 ):
    gameDisplay = textDisplay.NullGraphics()
    rules = ClassicGameRules(timeout)
    rules.quiet = True
    games = []

    for i in range( numGames ):
        beQuiet = i < numTraining
        game = rules.newGame( layout, pacman, ghosts, gameDisplay, True, catchExceptions)
        game.run()
        if not beQuiet: games.append(game)

        if record:
            import time, cPickle
            fname = ('recorded_games/recorded-game-%d' % (i + 1)) +  '-'.join([str(t) for t in time.localtime()[1:6]])
            f = file(fname, 'w')
            components = {'layout': layout, 'actions': game.moveHistory}
            cPickle.dump(components, f)
            f.close()

    if (numGames-numTraining) > 0:
        scores = [game.state.getScore() for game in games]
        wins = [game.state.isWin() for game in games]
        winRate = wins.count(True)/ float(len(wins))
        #HNere is where you propogate the error hum
        # oh well
        
        print 'Average Score:', sum(scores) / float(len(scores))
        #print 'Scores:       ', ', '.join([str(score) for score in scores])
        print 'Win Rate:      %d/%d (%.2f)' % (wins.count(True), len(wins), winRate)
        #print 'Record:       ', ', '.join([ ['Loss', 'Win'][int(w)] for w in wins])

    return games


# In[ ]:

# The model needs to be created twice (once for evaluation and once for training)
smart_agent_model.model = keras.models.load_model('models/smart_agent_model')
smart_agent_model.init_training(keras.models.load_model('models/smart_agent_model'))
smart_agent_model.model.save('models/smart_agent_model')
while True:
    runGames(**args)
    smart_agent_model.save('models/smart_agent_model') 


# In[ ]:


#Option 2: More fine grain control
# This would require me to rewrite the run method. This is ok too, just need a heads up. 


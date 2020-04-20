import numpy as np
from Constants import *
import random

def select_from_distribution(distribution):
    selection = random.random()
    total = 0.0
    for i in range(len(distribution)):
        total += distribution[i]
        if selection <= total:
            return i
    return None # This shouldn't happen
   
def convert_single_state(game_state):
    # See pacman.py for GameState and game.py for GameStateData
    
    walls = np.zeros((PLANE_WIDTH, PLANE_HEIGHT))
    state_walls = game_state.getWalls()
    for x in range(PLANE_WIDTH):
        for y in range(PLANE_HEIGHT):
            walls[x, y] = 1 if state_walls[x][y] else 0
            
    dots = np.zeros((PLANE_WIDTH, PLANE_HEIGHT))
    state_food = game_state.getFood()
    for x in range(PLANE_WIDTH):
        for y in range(PLANE_HEIGHT):
            dots[x, y] = 1 if state_food[x][y] else 0
            
    powerup = np.zeros((PLANE_WIDTH, PLANE_HEIGHT))
    state_capsules = game_state.getCapsules()
    for capsule in state_capsules:
        powerup[capsule[1], capsule[0]] = 1

    fruit = np.zeros((PLANE_WIDTH, PLANE_HEIGHT))
    
    pacman = np.zeros((PLANE_WIDTH, PLANE_HEIGHT))
    state_pacman_position = game_state.getPacmanPosition()
    powerup[state_pacman_position[0], state_pacman_position[1]] = 1
    state_ghost_positions = game_state.getGhostPositions()
    
    # Red Ghost
    blinky = np.zeros((PLANE_WIDTH, PLANE_HEIGHT))
    blinky[int(state_ghost_positions[0][0]), int(state_ghost_positions[0][1])] = 1

    # Pink Ghost
    pinky = np.zeros((PLANE_WIDTH, PLANE_HEIGHT))
    pinky[int(state_ghost_positions[1][0]), int(state_ghost_positions[1][1])] = 1

    # Blue Ghost
    inky = np.zeros((PLANE_WIDTH, PLANE_HEIGHT))
    inky[int(state_ghost_positions[2][0]), int(state_ghost_positions[2][1])] = 1

    # Orange Ghost
    clyde = np.zeros((PLANE_WIDTH, PLANE_HEIGHT))
    clyde[int(state_ghost_positions[3][0]), int(state_ghost_positions[3][1])] = 1

    return np.array([walls, dots, powerup, fruit, pacman, blinky, pinky, inky, clyde])

def convert_state_to_input(game_state, last_input):
    # See pacman.py for GameState and game.py for GameStateData
    result = np.zeros((TIMESTEP_PLANES*INPUT_TIMESTEPS, PLANE_WIDTH, PLANE_HEIGHT))
    result[0:TIMESTEP_PLANES] = convert_single_state(game_state)
    
    if last_input is None:
        # If this is the first input, copy the one for the current time step into the rest
        for i in range(INPUT_TIMESTEPS - 1):
            result[(INPUT_TIMESTEPS*(i+1)):(TIMESTEP_PLANES+INPUT_TIMESTEPS*(i+1))] = result[0:TIMESTEP_PLANES]
    else:
        # If this isn't the first input, use the last inputs as the input
        for i in range(INPUT_TIMESTEPS - 1):
            result[(INPUT_TIMESTEPS*(i+1)):(TIMESTEP_PLANES+INPUT_TIMESTEPS*(i+1))] = last_input[(INPUT_TIMESTEPS*i):(TIMESTEP_PLANES+INPUT_TIMESTEPS*i)]

        
    return result
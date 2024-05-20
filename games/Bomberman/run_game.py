import pygame

from game import GameState
from enums.algorithm import Algorithm

pygame.display.init()

def run_game():
    g = GameState()
    while True:
        reward, state, player, power_ups = g.step()
        #for p in power_ups:
        # for i in range(len(state)):
        #     for j in range(len(state[0])):
        #         print(state[j][i], end=" ")
        #     print()
        


if __name__ == "__main__":
    run_game()

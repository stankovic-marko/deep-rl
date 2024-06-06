import math
import pygame
import random
from gymnasium import Env
from gymnasium import spaces
import numpy as np
from copy import copy, deepcopy

from enums.power_up_type import PowerUpType
from enums.actions import Actions

from player import Player
from explosion import Explosion
from enemy import Enemy
from enums.algorithm import Algorithm
from power_up import PowerUp

FPS = 30
BACKGROUND_COLOR = (107, 142, 35)
GRID_BASE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

pygame.init()
INFO = pygame.display.Info()
TILE_SIZE = int(INFO.current_h * 0.035)
WINDOW_SIZE = (13 * TILE_SIZE, 13 * TILE_SIZE)
CLOCK = None
PLAYER_ALG = Algorithm.PLAYER
ENEMY_1_ALG = Algorithm.DIJKSTRA
ENEMY_2_ALG = Algorithm.DFS
ENEMY_3_ALG = Algorithm.DIJKSTRA
SHOW_PATH = False
SURFACE = pygame.display.set_mode(WINDOW_SIZE)
FPSCLOCK = pygame.time.Clock()


class Bombarder(Env):

    def __init__(self, render_mode=None):
        self.player = Player()
        self.enemy_list = []
        self.ene_blocks = []
        self.bombs = []
        self.explosions = []
        self.power_ups = []
        self.surface = SURFACE
        self.path = SHOW_PATH
        self.tile_size = TILE_SIZE
        self.scale = TILE_SIZE
        self.done = False
        self.cumulative_reward = 0.0
        self.render_mode = render_mode
        self.font = pygame.font.SysFont(None, 24)

        if ENEMY_1_ALG is not Algorithm.NONE:
            en1 = Enemy(11, 11, ENEMY_1_ALG)
            en1.load_animations('1', self.scale)
            self.enemy_list.append(en1)
            self.ene_blocks.append(en1)

        if ENEMY_2_ALG is not Algorithm.NONE:
            en2 = Enemy(1, 11, ENEMY_2_ALG)
            en2.load_animations('2', self.scale)
            self.enemy_list.append(en2)
            self.ene_blocks.append(en2)

        if ENEMY_3_ALG is not Algorithm.NONE:
            en3 = Enemy(11, 1, ENEMY_3_ALG)
            en3.load_animations('3', self.scale)
            self.enemy_list.append(en3)
            self.ene_blocks.append(en3)

        if PLAYER_ALG is Algorithm.PLAYER:
            self.player.load_animations(self.scale)
            self.ene_blocks.append(self.player)
        elif PLAYER_ALG is not Algorithm.NONE:
            en0 = Enemy(1, 1, PLAYER_ALG)
            en0.load_animations('', self.scale)
            self.enemy_list.append(en0)
            self.ene_blocks.append(en0)
            self.player.life = False
        else:
            self.player.life = False

        grass_img = pygame.image.load('images/terrain/grass.png')
        grass_img = pygame.transform.scale(grass_img, (self.scale, self.scale))

        block_img = pygame.image.load('images/terrain/block.png')
        block_img = pygame.transform.scale(block_img, (self.scale, self.scale))

        box_img = pygame.image.load('images/terrain/box.png')
        box_img = pygame.transform.scale(box_img, (self.scale, self.scale))

        bomb1_img = pygame.image.load('images/bomb/1.png')
        bomb1_img = pygame.transform.scale(bomb1_img, (self.scale, self.scale))

        bomb2_img = pygame.image.load('images/bomb/2.png')
        bomb2_img = pygame.transform.scale(bomb2_img, (self.scale, self.scale))

        bomb3_img = pygame.image.load('images/bomb/3.png')
        bomb3_img = pygame.transform.scale(bomb3_img, (self.scale, self.scale))

        explosion1_img = pygame.image.load('images/explosion/1.png')
        explosion1_img = pygame.transform.scale(
            explosion1_img, (self.scale, self.scale))

        explosion2_img = pygame.image.load('images/explosion/2.png')
        explosion2_img = pygame.transform.scale(
            explosion2_img, (self.scale, self.scale))

        explosion3_img = pygame.image.load('images/explosion/3.png')
        explosion3_img = pygame.transform.scale(
            explosion3_img, (self.scale, self.scale))

        self.terrain_images = [grass_img, block_img, box_img, grass_img]
        self.bomb_images = [bomb1_img, bomb2_img, bomb3_img]
        self.explosion_images = [explosion1_img,
                                 explosion2_img, explosion3_img]

        power_up_bomb_img = pygame.image.load('images/power_up/bomb.png')
        power_up_bomb_img = pygame.transform.scale(
            power_up_bomb_img, (self.scale, self.scale))

        power_up_fire_img = pygame.image.load('images/power_up/fire.png')
        power_up_fire_img = pygame.transform.scale(
            power_up_fire_img, (self.scale, self.scale))

        self.power_ups_images = [power_up_bomb_img, power_up_fire_img]
        self.state = [row[:] for row in GRID_BASE]

        self.observation_space = spaces.Box(
            0, 6, shape=(13, 13), dtype=np.int64)
        self.action_space = spaces.Discrete(6)

        generate_map(self.state)

    def step(self, action):
        pygame.event.pump()
        info = {}
        reward = 0.0

        standing = True
        if action != 5:
            standing = False

        # Update enemies
        for en in self.enemy_list:
            en.make_move(self.state, self.bombs, self.explosions,
                         self.ene_blocks, self.power_ups)

        px = self.player.pos_x//self.player.TILE_SIZE
        py = self.player.pos_y//self.player.TILE_SIZE
        # Punish if trapped
        if self.state[px - 1][py] != 0 and self.state[px + 1][py] != 0 and self.state[px][py - 1] != 0 and self.state[px][py + 1] != 0:
            reward -= 0.2

        # Bomb vicinity
        for bomb in self.bombs:

            distance = math.sqrt((bomb.pos_x-px)**2+(
                bomb.pos_y-py)**2)
            if distance > bomb.range:  # Skip check if not in bomb range
                continue

            if [px, py] in bomb.sectors:
                reward -= 0.5  # Punish if in bomb sector

            # if bomb.pos_x == px or bomb.pos_y == py:
            #     reward -= 0.5  # Punish for standing on bomb path
            # else:
            #     if standing:
            #         reward += 0.2  # Reward for standing on safety near bomb
            # if bomb.pos_x == px and bomb.pos_y == py:
            #     reward -= 1  # Punish for standing on top of the bomb

        # Explosion vicinity
        for explosion in self.explosions:
            distance = math.sqrt((explosion.sourceX-px)**2+(
                explosion.sourceY-py)**2)
            if distance > explosion.range:  # Skip check if not in explosion range
                continue
            if [px, py] in explosion.sectors:
                reward -= 0.5  # Punish if in explosion sector
            # else:
            #     if standing:
            #         reward += 0.2  # Reward for standing on safety near explosion

        # Handle player actions if the player is alive
        if self.player.life:

            enemies_dead = all([not en.life for en in self.enemy_list])
            if enemies_dead:
                self.done = True

            movement = False
            direction = self.player.direction
            if action == 0:  # Move up
                movement = self.player.move(
                    0, -1, self.state, self.ene_blocks, self.power_ups)
                direction = 0
            elif action == 1:  # Move down
                movement = self.player.move(
                    0, 1, self.state, self.ene_blocks, self.power_ups)
                direction = 1
            elif action == 2:  # Move left
                movement = self.player.move(-1, 0, self.state,
                                            self.ene_blocks, self.power_ups)
                direction = 2
            elif action == 3:  # Move right
                movement = self.player.move(
                    1, 0, self.state, self.ene_blocks, self.power_ups)
                direction = 3
            elif action == 4:  # Plant bomb
                if self.player.bomb_limit > 0:
                    temp_bomb = self.player.plant_bomb(self.state)
                    self.bombs.append(temp_bomb)
                    self.state[temp_bomb.pos_x][temp_bomb.pos_y] = 3
                    self.player.bomb_limit -= 1
                    # reward += 0.1  # Reward for planting a bomb

            self.player.direction = direction

            # Reward for placing all the bombs
            # if self.player.bomb_limit == 0:
            #     reward += 0.05

            for bomb in self.bombs:
                if isinstance(bomb.bomber, Player):
                    reward += 0.035  # Small reward for each bomb placed

            if movement:
                reward += 0.01  # Small reward for moving

            # Additional rewards or penalties based on power-ups and other factors
            reward += self.player.num_power_ups * 0.075

            # Reward for killing each enemy
            reward += (0.25 * self.player.kills)

            # Reward for destroying crates
            reward += (0.03 * self.player.crates_destroyed)
            # Punish if enemies are faster at destoying crates
            # for enemy in self.enemy_list:
            #     reward -= (0.05 * enemy.crates_destroyed)
        else:
            self.done = True
            reward -= 0.3  # Penalty for player death

        # # Update rewards for enemies
        # for enemy in self.enemy_list:
        #     if not enemy.life:
        #         reward += 0.5  # Reward for each enemy killed
        #     else:
        #         reward -= 0.01  # Small penalty for each enemy still alive

        self.cumulative_reward = reward

        if self.render_mode == "human":
            self.render()

        # Update bombs
        dt = FPSCLOCK.tick(FPS)
        self.update_bombs(dt)

        # Update game state
        updated_state = deepcopy(self.state)
        # Add player position to state
        updated_state[self.player.pos_x // 4][self.player.pos_y // 4] = 5
        for explosion in self.explosions:
            updated_state[explosion.sourceX][explosion.sourceY] = 3
        for enemy in self.enemy_list:
            updated_state[enemy.pos_x//4][enemy.pos_y//4] = 4
        # ********************************************************************
        for explosion in self.explosions:
            for sector in explosion.sectors:
                updated_state[sector[0]][sector[1]] = 3
        # ********************************************************************
        for power_up in self.power_ups:
            updated_state[power_up.pos_x][power_up.pos_y] = 6
        # ********************************************************************

        return np.array(updated_state), reward, self.done, False, info

    def reset(self, *, seed=None, options=None):
        self.seed = seed
        self.options = options
        info = {}
        self.__init__(render_mode=self.render_mode)
        return np.array(self.state), info

    def render(self):
        self.surface.fill(BACKGROUND_COLOR)
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                self.surface.blit(self.terrain_images[self.state[i][j]], (
                    i * self.tile_size, j * self.tile_size, self.tile_size, self.tile_size))

        for pu in self.power_ups:
            self.surface.blit(self.power_ups_images[pu.type.value], (
                pu.pos_x * self.tile_size, pu.pos_y * self.tile_size, self.tile_size, self.tile_size))

        for x in self.bombs:
            self.surface.blit(self.bomb_images[x.frame], (
                x.pos_x * self.tile_size, x.pos_y * self.tile_size, self.tile_size, self.tile_size))

        for y in self.explosions:
            for x in y.sectors:
                self.surface.blit(self.explosion_images[y.frame], (
                    x[0] * self.tile_size, x[1] * self.tile_size, self.tile_size, self.tile_size))
        if self.player.life:
            self.surface.blit(self.player.animation[self.player.direction][self.player.frame],
                              (self.player.pos_x * (self.tile_size / 4), self.player.pos_y * (self.tile_size / 4), self.tile_size, self.tile_size))
        for en in self.enemy_list:
            if en.life:
                self.surface.blit(en.animation[en.direction][en.frame],
                                  (en.pos_x * (self.tile_size / 4), en.pos_y * (self.tile_size / 4), self.tile_size, self.tile_size))

        # Display info on screen
        reward = self.font.render(
            f'K:{self.player.kills} C:{self.player.crates_destroyed} P:{self.player.num_power_ups} R:{self.cumulative_reward:6.4f}', True, (255, 0, 0))
        self.surface.blit(reward, (0, 0))
        pygame.display.update()

    def update_bombs(self, dt):
        for b in self.bombs:
            b.update(dt)
            if b.time < 1:
                b.bomber.bomb_limit += 1
                self.state[b.pos_x][b.pos_y] = 0
                exp_temp = Explosion(b.pos_x, b.pos_y, b.range)
                exp_temp.explode(self.state, self.bombs, b, self.power_ups)
                exp_temp.clear_sectors(self.state, random, self.power_ups)
                self.explosions.append(exp_temp)
        if self.player not in self.enemy_list:
            self.player.check_death(self.explosions)
        for en in self.enemy_list:
            en.check_death(self.explosions)
        for e in self.explosions:
            e.update(dt)
            if e.time < 1:
                self.explosions.remove(e)


def generate_map(grid):
    for i in range(1, len(grid) - 1):
        for j in range(1, len(grid[i]) - 1):
            if grid[i][j] != 0:
                continue
            elif (i < 3 or i > len(grid) - 4) and (j < 3 or j > len(grid[i]) - 4):
                continue
            if random.randint(0, 9) < 7:
                grid[i][j] = 2

    return

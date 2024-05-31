import pygame
import sys
import random
from copy import copy, deepcopy

from enums.power_up_type import PowerUpType
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


class GameState:

    def __init__(self):
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
        self.game_ended = False

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
        self.grid = [row[:] for row in GRID_BASE]
        generate_map(self.grid)

    def step(self):
        reward = 0.1
        for en in self.enemy_list:
            en.make_move(self.grid, self.bombs, self.explosions,
                         self.ene_blocks, self.power_ups)

        if self.player.life:
            keys = pygame.key.get_pressed()
            temp = self.player.direction
            movement = False
            if keys[pygame.K_DOWN]:
                temp = 0
                self.player.move(
                    0, 1, self.grid, self.ene_blocks, self.power_ups)
                movement = True
            elif keys[pygame.K_RIGHT]:
                temp = 1
                self.player.move(
                    1, 0, self.grid, self.ene_blocks, self.power_ups)
                movement = True
            elif keys[pygame.K_UP]:
                temp = 2
                self.player.move(0, -1, self.grid,
                                 self.ene_blocks, self.power_ups)
                movement = True
            elif keys[pygame.K_LEFT]:
                temp = 3
                self.player.move(-1, 0, self.grid,
                                 self.ene_blocks, self.power_ups)
                movement = True
            if temp != self.player.direction:
                self.player.frame = 0
                self.player.direction = temp
            if movement:
                if self.player.frame == 2:
                    self.player.frame = 0
                else:
                    self.player.frame += 1
        else:
            self.__init__()

        for enemy in self.enemy_list:
            if not enemy.life:
                reward += 0.05
            else:
                reward -= enemy.num_power_ups * 0.01
        reward += self.player.num_power_ups * 0.01

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                sys.exit(0)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    if self.player.bomb_limit == 0 or not self.player.life:
                        continue
                    temp_bomb = self.player.plant_bomb(self.grid)
                    self.bombs.append(temp_bomb)
                    self.grid[temp_bomb.pos_x][temp_bomb.pos_y] = 3
                    self.player.bomb_limit -= 1

        dt = FPSCLOCK.tick(30)
        self.update_bombs(dt)
        self.draw()
        table = deepcopy(self.grid)
        table[self.player.pos_x//4][self.player.pos_y//4] = 5
        return reward, self.grid, self.player, self.power_ups

    def draw(self):
        self.surface.fill(BACKGROUND_COLOR)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                self.surface.blit(self.terrain_images[self.grid[i][j]], (
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

        if self.game_ended:
            #tf = font.render("Press ESC to go back to menu", False, (153, 153, 255))
            #s.blit(tf, (10, 10))
            print("player died")

        pygame.display.update()

    def update_bombs(self, dt):
        for b in self.bombs:
            b.update(dt)
            if b.time < 1:
                b.bomber.bomb_limit += 1
                self.grid[b.pos_x][b.pos_y] = 0
                exp_temp = Explosion(b.pos_x, b.pos_y, b.range)
                exp_temp.explode(self.grid, self.bombs, b, self.power_ups)
                exp_temp.clear_sectors(self.grid, random, self.power_ups)
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

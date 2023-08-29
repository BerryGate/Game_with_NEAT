"""
Agent playing game using NEAT
This file showcases the training process

GitHub Presentation
August 2023
"""

import pygame as pg
import random
import os
import neat
import math
import time
import pickle

pg.font.init()
pg.init()

WIN_W, WIN_H = 1400, 800
WIN = pg.display.set_mode((WIN_W, WIN_H))
font = pg.font.SysFont('conmicsans', 30)
gen = 0

class Ball:
    def __init__(self):
        self.x = random.randint(100, 1300)
        self.y = random.randint(100, 700)
        self.radius = 17.5
        self.SPEED = 4
        self.color = tuple([random.randint(0, 255) for i in range(3)])

    def draw(self, win):
        pg.draw.circle(win, self.color, (self.x, self.y), self.radius, 0)

    def move(self, output):
        if output.index(max(output)) == 0:
            self.x += self.SPEED
        elif output.index(max(output)) == 1:
            self.x -= self.SPEED
        elif output.index(max(output)) == 2:
            self.y -= self.SPEED
        elif output.index(max(output)) == 3:
            self.y += self.SPEED


class Food:
    def __init__(self, color):
        self.x = random.randint(50, 1350)
        self.y = random.randint(20, 750)
        self.length = 12.5
        self.color = color

    def draw(self, win):
        pg.draw.rect(win, self.color, (self.x, self.y, self.length, self.length))

    def is_consumed(self, ball):
        if ball.x - ball.radius <= self.x + 0.5 * self.length <= ball.x + ball.radius and ball.y - ball.radius <= self.y + 0.5 * self.length <= ball.y + ball.radius:
            return True
        return False


def wall_collision(ball):
    if ball.x - ball.radius <= 0 or ball.x + ball.radius >= WIN_W or ball.y - ball.radius <= 0 or ball.y + ball.radius >= WIN_H:
        return True


def master_draw(balls, foods, gen, alive):
    WIN.fill((40, 75, 200))
    for ball in balls:
        ball.draw(WIN)
    for food in foods:
        food.draw(WIN)
    gen_text = font.render(f'# Generation: {gen}', True, (100, 200, 100))
    WIN.blit(gen_text, (5, 10))
    alive_text = font.render(f'# Alive: {alive}', True, (100, 200, 100))
    WIN.blit(alive_text, (5, 40))
    pg.display.update()


def main(genomes, config):
    global gen
    gen += 1

    nets = [neat.nn.FeedForwardNetwork.create(genome[1], config) for genome in genomes]
    balls = [Ball() for i in range(len(genomes))]
    foods = [Food(balls[i].color) for i in range(len(genomes))]
    distance_to_food = [math.sqrt(((balls[i].x - foods[i].x) ** 2) + ((balls[i].y - foods[i].y) ** 2)) for i in range(len(balls))]
    for genome in genomes:
        genome[1].fitness = 0

    fps = 30
    clock = pg.time.Clock()

    previous_distance = [(ball.x, ball.y) for ball in balls]
    last_time = time.time()
    clean_time = time.time()

    while len(balls) > 0:
        clock.tick(fps)
        del_lis = []
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        for i in range(len(balls)):
            if wall_collision(balls[i]):
                genomes[i][1].fitness -= 7
                del_lis.append(i)
                continue

            if time.time() >= last_time + 1.75 and abs(previous_distance[i][0] - balls[i].x) < 15 and abs(previous_distance[i][1] - balls[i].y) < 15:
                # If not moving for 1.75 seconds
                genomes[i][1].fitness -= 7
                del_lis.append(i)
                continue

            if time.time() >= clean_time + 52.5:
                del_lis.append(i)
                continue

            if foods[i].is_consumed(balls[i]):
                foods[i] = Food(balls[i].color)
                balls[i].radius += 0.5
                genomes[i][1].fitness += 3

            current_distance_to_food = math.sqrt(((balls[i].x - foods[i].x) ** 2) + ((balls[i].y - foods[i].y) ** 2))
            if current_distance_to_food < distance_to_food[i]:
                genomes[i][1].fitness += 0.04 # Adding 1 per second
            else:
                genomes[i][1].fitness -= 0.0333
            distance_to_food[i] = current_distance_to_food

            output = nets[i].activate((balls[i].x, balls[i].y, current_distance_to_food, balls[i].x - foods[i].x, balls[i].y - foods[i].y, WIN_W - balls[i].x, WIN_H - balls[i].y))
            balls[i].move(output)

        for ind in sorted(del_lis, reverse=True):
            for lis in [nets, genomes, foods, balls, distance_to_food, previous_distance]:
                del lis[ind]

        if time.time() >= last_time + 1.75:
            last_time = time.time()
            previous_distance = [(ball.x, ball.y) for ball in balls]

        if time.time() >= clean_time + 52.5:
            clean_time = time.time()

        master_draw(balls, foods, gen, len(balls))


def save_genomes(best_genomes):
    with open("progress_3.pkl", "wb") as file:
        pickle.dump(best_genomes, file)
        file.close()


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config_training.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    best_genomes = population.run(main, 50)
    save_genomes(best_genomes)



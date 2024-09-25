import pygame
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model
import multiprocessing as mp
import random

# Function to create a new neural network model
def create_model():
    model = Sequential([
        Flatten(input_shape=(5,)),  # Input: ball_x, ball_y, ball_speed_x_sign, ball_speed_y_sign, paddle_y
        Dense(24, activation='relu'),  # Increased neurons for better learning
        Dense(24, activation='relu'),  # Increased neurons for better learning
        Dense(3, activation='softmax')  # Output: up, down, stay
    ])
    return model



BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
def initialize_population(pop_size):
    population_weights = []
    for _ in range(pop_size):
        model = create_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Randomly initialize weights and biases for each Dense layer
        for layer in model.layers:
            if isinstance(layer, Dense):
                weights = layer.get_weights()
                weights[0] = np.random.uniform(-1.0, 1.0, size=weights[0].shape)  # Random weights
                weights[1] = np.random.uniform(-1.0, 1.0, size=weights[1].shape)  # Random biases
                layer.set_weights(weights)

        # Save the weights instead of the model to avoid pickling issues
        population_weights.append(model.get_weights())
    return population_weights

# Define the Paddle class
class Paddle(pygame.sprite.Sprite):
    def __init__(self, x_pos, ai=False, model=None):
        super().__init__()
        self.image = pygame.Surface((10, 100))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x_pos
        self.rect.y = 250
        self.speed = 5
        self.ai = ai
        self.model = model

    def update(self, keys, up_key=None, down_key=None, ball=None):
        if self.ai and ball:
            # Get the current state
            state = np.array([ball.rect.x, ball.rect.y, ball.speed_x, ball.speed_y, self.rect.y])
            state = state.reshape(1, 5)

            # Predict action
            action = np.argmax(self.model.predict(state))

            # Move the paddle based on the action
            if action == 0:
                self.rect.y -= self.speed
            elif action == 1:
                self.rect.y += self.speed
        else:
            if keys[up_key]:
                self.rect.y -= self.speed
            if keys[down_key]:
                self.rect.y += self.speed

        # Keep paddle within screen bounds
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > 600:
            self.rect.bottom = 600

# Rest of the Pong game code remains the same...
class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((10, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (400, 300)
        self.speed_x = 5
        self.speed_y = 5
        self.insp_x = 5
        self.insp_y = 5
        self.bounce_count = 0

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # Bounce off top and bottom walls
        if self.rect.top <= 0 or self.rect.bottom >= 600:
            self.speed_y = -self.speed_y
            self.bounce_count += 1

        # Check for left and right out of bounds
        if self.rect.left <= 0:
            return "left"
        if self.rect.right >= 800:
            return "right"
        if self.bounce_count == 5:
            self.increase_speed()
            self.bounce_count = 0

        return None

    def reset(self):
        self.rect.center = (400, 300)
        self.speed_x = -self.insp_x  # Change ball direction
        self.speed_y = 5 if self.speed_y > 0 else -5
        self.bounce_count = 0

    def increase_speed(self):
        self.speed_x *= 1.2
        self.speed_y *= 1.2


# Evaluate the fitness of a model based on the number of bounces of the left paddle
def evaluate_fitness(model_weights, generation, model_count):
    # Load the expert model within the function to avoid multiprocessing issues
    expert_model = load_model('pong_ai_model2.h5')

    # Create the model and set its weights
    model = create_model()
    model.set_weights(model_weights)

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Pong Game")

    # Corrected paddle positions: left paddle on the left, right paddle on the right
    left_paddle = Paddle(30, ai=True, model=model)
    right_paddle = Paddle(760, ai=True, model=expert_model)
    ball = Ball()

    all_sprites = pygame.sprite.Group()
    all_sprites.add(left_paddle, right_paddle, ball)

    clock = pygame.time.Clock()
    left_bounces = 0  # Track the number of successful bounces by the left paddle

    font = pygame.font.Font(None, 36)
    bounce = 0
    for frame in range(1000):  # Run the game for a fixed number of frames
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        left_paddle.update(keys, ball=ball)
        right_paddle.update(keys, ball=ball)
        out_of_bounds = ball.update()

        # Check for ball bounce off the left paddle
        if pygame.sprite.collide_rect(ball, left_paddle):
            ball.speed_x = -ball.speed_x
            left_bounces += 1  # Increment the number of bounces when the left paddle hits the ball
            bounce += 1

        # Check for ball bounce off the right paddle
        if pygame.sprite.collide_rect(ball, right_paddle):
            ball.speed_x = -ball.speed_x
            bounce += 1

        # Handle ball going out of bounds
        if out_of_bounds == "left":
            ball.reset()
            left_bounces -= 1  # Penalize for missing the ball
            bounce = 0
        elif out_of_bounds == "right":
            ball.reset()
            bounce = 0
        if bounce == 5:
            ball.increase_speed()
            bounce = 0
        screen.fill((0, 0, 0))  # BLACK color
        all_sprites.draw(screen)


        round_text = font.render(f"Generation: {generation + 1}", True, (255, 255, 255))  # WHITE color
        screen.blit(round_text, (10, 10))
        model_text = font.render(f"Models Trained: {model_count + 1}", True, (255, 255, 255))  # WHITE color
        screen.blit(model_text, (10, 50))
        fitness_text = font.render(f"Fitness: {left_bounces}", True, (255, 255, 255))  # WHITE color

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return left_bounces  # The fitness score is now the number of bounces by the left paddle

# Function to select parents based on fitness
def select_parents(population_weights, fitnesses, num_parents):
    # Select the best models based on fitness
    parents_weights = [population_weights[i] for i in np.argsort(fitnesses)[-num_parents:]]
    return parents_weights

# Function to perform crossover between two parent models
def crossover(parent1_weights, parent2_weights):
    child_weights = []
    for w1, w2 in zip(parent1_weights, parent2_weights):
        # Simple crossover at the weight level
        mask = np.random.rand(*w1.shape) > 0.5
        w_new = np.where(mask, w1, w2)
        child_weights.append(w_new)
    return child_weights

# Function to mutate a model's weights
def mutate(weights, mutation_rate=0.1):
    new_weights = []
    for w in weights:
        mutation = mutation_rate * np.random.randn(*w.shape)
        w_new = w + mutation
        new_weights.append(w_new)
    return new_weights

# Function to evaluate a model's fitness in parallel
def evaluate_model_parallel(args):
    model_weights, generation, model_count = args
    fitness = evaluate_fitness(model_weights, generation, model_count)
    return model_weights, fitness

# Genetic algorithm for training models with parallel evaluation
def genetic_algorithm_parallel(pop_size, num_generations, num_parents, mutation_rate):
    population_weights = initialize_population(pop_size)

    for generation in range(num_generations):
        fitnesses = []
        # Use a multiprocessing pool for parallel evaluation
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(
                evaluate_model_parallel,
                [(weights, generation, i) for i, weights in enumerate(population_weights)]
            )

        # Extract fitness values and assign them to models
        population_weights = []
        fitnesses = []
        for weights, fitness in results:
            population_weights.append(weights)
            fitnesses.append(fitness)

        # Select the best parents
        parents_weights = select_parents(population_weights, fitnesses, num_parents)

        # Generate next generation
        next_population_weights = []
        for i in range(pop_size):
            parent1_weights, parent2_weights = random.sample(parents_weights, 2)
            child_weights = crossover(parent1_weights, parent2_weights)
            child_weights = mutate(child_weights, mutation_rate)
            next_population_weights.append(child_weights)

        population_weights = next_population_weights
        print(f"Generation {generation + 1} - Best Fitness: {max(fitnesses)}")

    return population_weights

# Save the best model from the final population
def save_best_model(population_weights):
    best_model = create_model()
    best_model.set_weights(population_weights)
    best_model.save('best_pong_ai_model.h5')

# Main function to run the genetic algorithm and save the best model
if __name__ == "__main__":
    pop_size = 10  # Increased population size
    num_generations = 30  # Reduced for testing purposes
    num_parents = 10
    mutation_rate = 0.1 # Increased mutation rate for diversity

    mp.set_start_method('spawn', force=True)

    best_population_weights = genetic_algorithm_parallel(pop_size, num_generations, num_parents, mutation_rate)

    # Find the best model in the final population
    best_fitness = -1
    best_weights = None
    for i, weights in enumerate(best_population_weights):
        fitness = evaluate_fitness(weights, num_generations, i)
        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = weights

    save_best_model(best_weights)

import pygame
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model
import multiprocessing as mp

# Load the expert model
expert_model = load_model('pong_ai_model.h5')
expert_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to create a new neural network model
def create_model():
    model = Sequential([
        Flatten(input_shape=(5,)),  # Input: ball_x, ball_y, ball_speed_x, ball_speed_y, paddle_y
        Dense(24, activation='relu'),  # Increased neurons for better learning
        Dense(2, activation='softmax')  # Output: up, down
    ])
    return model

# Function to initialize a population of models with random initial weights
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        model = create_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Randomly initialize weights and biases for each Dense layer
        for layer in model.layers:
            if isinstance(layer, Dense):
                weights = layer.get_weights()
                weights[0] = np.random.uniform(-1.0, 1.0, size=weights[0].shape)  # Random weights
                weights[1] = np.random.uniform(-1.0, 1.0, size=weights[1].shape)  # Random biases
                layer.set_weights([weights[0], weights[1]])

        population.append(model)
    return population

# Define the Paddle class
class Paddle(pygame.sprite.Sprite):
    def __init__(self, x_pos, ai=False, model=None):
        super().__init__()
        self.image = pygame.Surface((10, 100))
        self.image.fill((255, 255, 255))  # WHITE color
        self.rect = self.image.get_rect()
        self.rect.x = x_pos
        self.rect.y = 250
        self.speed = 5
        self.ai = ai
        self.model = model

    def update(self, keys, up_key=None, down_key=None, ball=None):
        if self.ai and ball:
            # Get the current state (5 inputs: ball_x, ball_y, ball_speed_x, ball_speed_y, paddle_y)
            state = np.array([ball.rect.x / 800, ball.rect.y / 600, ball.speed_x / abs(ball.speed_x), ball.speed_y / abs(ball.speed_y), self.rect.y / 600])
            state = state.reshape(1, 5)

            probabilities = self.model.predict(state)
            action = np.argmax(probabilities)

            # Debugging action output
            print(f"State: {state}, Probabilities: {probabilities}, Action: {action}")

            # Move paddle based on action: 0 - up, 1 - down
            if action == 0:  # Move up
                self.rect.y -= self.speed
            elif action == 1:  # Move down
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

# Define the Ball class
class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((10, 10))
        self.image.fill((255, 255, 255))  # WHITE color
        self.rect = self.image.get_rect()
        self.rect.center = (400, 300)
        self.speed_x = 5
        self.speed_y = 5

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # Bounce off top and bottom walls
        if self.rect.top <= 0 or self.rect.bottom >= 600:
            self.speed_y = -self.speed_y

        # Check for left and right out of bounds
        if self.rect.left <= 0:
            return "right"
        if self.rect.right >= 800:
            return "left"

        return None

    def reset(self):
        self.rect.center = (400, 300)
        self.speed_x = -self.speed_x  # Change ball direction
        self.speed_y = 5 if self.speed_y > 0 else -5

# Evaluate the fitness of a model based on the number of bounces of the left paddle
def evaluate_fitness(model, generation, model_count):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Pong Game")

    right_paddle = Paddle(30, ai=True, model=expert_model) 
    left_paddle = Paddle(760, ai=True, model=model)
    ball = Ball()

    all_sprites = pygame.sprite.Group()
    all_sprites.add(left_paddle, right_paddle, ball)

    clock = pygame.time.Clock()
    left_bounces = 0  # Track the number of successful bounces by the left paddle

    font = pygame.font.Font(None, 36)

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

        # Handle ball going out of bounds
        if out_of_bounds == "left":
            ball.reset()
        elif out_of_bounds == "right":
            ball.reset()

        screen.fill((0, 0, 0))  # BLACK color
        all_sprites.draw(screen)

        # Display generation and model count
        round_text = font.render(f"Generation: {generation + 1}", True, (255, 255, 255))  # WHITE color
        screen.blit(round_text, (10, 10))
        model_text = font.render(f"Models Trained: {model_count + 1}", True, (255, 255, 255))  # WHITE color
        screen.blit(model_text, (10, 50))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return left_bounces  # The fitness score is now the number of bounces by the left paddle

# Function to select parents based on fitness
def select_parents(population, fitnesses, num_parents):
    parents = [population[i] for i in np.argsort(fitnesses)[-num_parents:]]
    return parents

# Function to perform crossover between two parent models
def crossover(parent1, parent2):
    child = create_model()
    for layer in range(len(parent1.layers)):
        if isinstance(parent1.layers[layer], Dense):
            weights1 = parent1.layers[layer].get_weights()
            weights2 = parent2.layers[layer].get_weights()
            new_weights = [0.5 * w1 + 0.5 * w2 for w1, w2 in zip(weights1, weights2)]
            child.layers[layer].set_weights(new_weights)
    return child

# Function to mutate a model
def mutate(model, mutation_rate=0.1):  # Slightly increased mutation rate
    for layer in model.layers:
        if isinstance(layer, Dense):
            weights = layer.get_weights()
            weights = [w + mutation_rate * np.random.randn(*w.shape) for w in weights]
            layer.set_weights(weights)

# Function to evaluate a model's fitness in parallel
def evaluate_model_parallel(args):
    model, generation, model_count = args
    fitness = evaluate_fitness(model, generation, model_count)
    return model, fitness

# Genetic algorithm for training models with parallel evaluation
def genetic_algorithm_parallel(pop_size, num_generations, num_parents, mutation_rate):
    population = initialize_population(pop_size)

    for generation in range(num_generations):
        fitnesses = []
        # Use a multiprocessing pool for parallel evaluation
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(evaluate_model_parallel, [(model, generation, i) for i, model in enumerate(population)])
        
        # Extract fitness values and assign them to models
        for model, fitness in results:
            fitnesses.append(fitness)
            model.fitness = fitness

        # Select the best parents
        parents = select_parents(population, fitnesses, num_parents)

        # Generate next generation
        next_population = []
        for i in range(pop_size):
            parent1, parent2 = np.random.choice(parents, 2)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            next_population.append(child)
        
        population = next_population
        print(f"Generation {generation + 1} - Best Fitness: {max(fitnesses)}")
    
    return population

# Save the final population's best model
def save_best_model(population):
    best_model = max(population, key=lambda model: model.fitness)
    best_model.save('best_pong_ai_model.h5')

# Main function to run the genetic algorithm and save the best model
if __name__ == "__main__":
    pop_size = 50  # Increased population size
    num_generations = 100  # Increased number of generations
    num_parents = 10
    mutation_rate = 0.1  # Increased mutation rate

    best_population = genetic_algorithm_parallel(pop_size, num_generations, num_parents, mutation_rate)
    save_best_model(best_population)

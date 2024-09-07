import pygame
import sys
import numpy as np
from tensorflow.keras.models import load_model

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Pong Game")
pygame.init()
pygame.font.init()
# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
# Load the trained model
model = load_model('pong_ai_model.h5')

class Paddle(pygame.sprite.Sprite):
    def __init__(self, x_pos, ai=False):
        super().__init__()
        self.image = pygame.Surface((10, 100))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x_pos
        self.rect.y = 250
        self.speed = 5
        self.ai = ai

    def update(self, keys, up_key=None, down_key=None, ball=None):
        if self.ai and ball:
            # Get the current state
            state = np.array([ball.rect.x, ball.rect.y, ball.speed_x, ball.speed_y, self.rect.y])
            state = state.reshape(1, 5)

            # Predict action
            action = np.argmax(model.predict(state))

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

# Create the paddles and ball
left_paddle = Paddle(30)
right_paddle = Paddle(760, ai=True)  # Right paddle with AI
ball = Ball()

all_sprites = pygame.sprite.Group()
all_sprites.add(left_paddle, right_paddle, ball)

# Initialize scores
left_score = 0
right_score = 0

# Set up font for the scoreboard
font = pygame.font.Font(None, 74)

def draw_score():
    left_text = font.render(str(left_score), True, WHITE)
    right_text = font.render(str(right_score), True, WHITE)
    screen.blit(left_text, (320, 10))
    screen.blit(right_text, (420, 10))

def end_game(left_score, right_score):
    if left_score == 3:
        end_text = font.render("Player 1 Wins!", True, WHITE)
        return end_text
    if right_score == 3:
        end_text = font.render("Player 2 Wins!", True, WHITE)
        return end_text
    return None

def reset_game():
    global left_score, right_score, ball
    left_score = 0
    right_score = 0
    ball.reset()

# Game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the current state of all keyboard buttons
    keys = pygame.key.get_pressed()

    # Update game state
    left_paddle.update(keys, pygame.K_w, pygame.K_s)
    right_paddle.update(keys, ball=ball)
    out_of_bounds = ball.update()

    # Check for paddle collisions
    if pygame.sprite.collide_rect(ball, left_paddle) or pygame.sprite.collide_rect(ball, right_paddle):
        ball.speed_x = -ball.speed_x

    # Update scores and reset ball
    if out_of_bounds == "left":
        right_score += 1
        ball.reset()
    elif out_of_bounds == "right":
        left_score += 1
        ball.reset()

    # Check for end game condition
    winner_message = end_game(left_score, right_score)
    if winner_message:
        # Clear the screen and display the winning message
        screen.fill(BLACK)
        screen.blit(winner_message, (200, 200))
        # Display restart message
        restart_font = pygame.font.Font(None, 36)
        restart_text = restart_font.render("Wanna play again? Press 'r' to restart", True, WHITE)
        screen.blit(restart_text, (200, 300))
        pygame.display.flip()

        # Wait for 'r' key press to restart the game
        waiting_for_restart = True
        while waiting_for_restart:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_restart = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    reset_game()
                    waiting_for_restart = False
    else:
        # Draw everything
        screen.fill(BLACK)
        all_sprites.draw(screen)
        draw_score()

        # Update the display
        pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)

pygame.quit()
sys.exit()

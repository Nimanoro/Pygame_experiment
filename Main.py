import pygame
import sys
import numpy as np
import os

# Initialize Pygame
pygame.init()

# Set up the game window
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Pong Game")

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Define the Paddle class
class Paddle(pygame.sprite.Sprite):
    def __init__(self, x_pos):
        super().__init__()
        self.image = pygame.Surface((10, 100))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x_pos
        self.rect.y = 250
        self.speed = 5

    def update(self, keys, up_key, down_key):
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
            return "right"
        if self.rect.right >= 800:
            return "left"
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

# Create the paddles and ball
left_paddle = Paddle(30)
right_paddle = Paddle(760)
ball = Ball()

all_sprites = pygame.sprite.Group()
all_sprites.add(left_paddle, right_paddle, ball)

# Initialize scores
left_score = 0
right_score = 0

# Set up font for the scoreboard
font = pygame.font.Font(None, 74)
button_font = pygame.font.Font(None, 36)

# Data collection variables
data = []
labels = []
save_data = False

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

def draw_button(text, rect, color):
    pygame.draw.rect(screen, color, rect)
    text_surf = button_font.render(text, True, WHITE)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

# Load existing data if available
if os.path.exists('pong_data.npy') and os.path.exists('pong_labels.npy'):
    data = np.load('pong_data.npy').tolist()
    labels = np.load('pong_labels.npy').tolist()

# Button rects
save_button_rect = pygame.Rect(300, 400, 200, 60)

# Game loop
running = True
waiting_for_save = False
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and waiting_for_save:
            if save_button_rect.collidepoint(event.pos):
                save_data = True
                running = False

    # Get the current state of all keyboard buttons
    keys = pygame.key.get_pressed()

    if not waiting_for_save:
        # Update game state
        left_paddle.update(keys, pygame.K_w, pygame.K_s)
        right_paddle.update(keys, pygame.K_UP, pygame.K_DOWN)
        out_of_bounds = ball.update()

        # Collect data: [ball_x, ball_y, ball_speed_x, ball_speed_y, paddle_y]
        state = [ball.rect.x, ball.rect.y, ball.speed_x, ball.speed_y, right_paddle.rect.y]

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

            # Set flag to show save button after game ends
            waiting_for_save = True
        else:
            # Determine the action (up, down, stay) for the right paddle
            if keys[pygame.K_UP]:
                action = [1, 0, 0]  # Up
            elif keys[pygame.K_DOWN]:
                action = [0, 1, 0]  # Down
            else:
                action = [0, 0, 1]  # Stay

            # Collect data
            data.append(state)
            labels.append(action)

            # Draw everything
            screen.fill(BLACK)
            all_sprites.draw(screen)
            draw_score()

            # Update the display
            pygame.display.flip()
    else:
        # Draw save data button
        screen.fill(BLACK)
        draw_button("Save Data", save_button_rect, GREEN)
        pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)

# Save the collected data if save_data is True
if save_data:
    np.save('pong_data.npy', np.array(data))
    np.save('pong_labels.npy', np.array(labels))

pygame.quit()
sys.exit()

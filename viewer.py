import pygame
import sys
import numpy as np

ROWS = 6
COLS = 7
CELL_SIZE = 100

WIDTH = COLS * CELL_SIZE
HEIGHT = (ROWS + 1) * CELL_SIZE

BLUE = (40, 80, 220)
BLACK = (0, 0, 0)
WHITE = (240, 240, 240)
RED = (220, 50, 50)
YELLOW = (240, 220, 70)


class Connect4Viewer:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Connect-4 Tournament")

        self.font = pygame.font.SysFont("Arial", 36)
        self.legend_font = pygame.font.SysFont("Arial", 24)
        self.red_player = ""
        self.yellow_player = ""

    def set_players(self, red_name: str, yellow_name: str):
        self.red_player = red_name
        self.yellow_player = yellow_name

    def _draw_legend(self):
        if not self.red_player:
            return
        radius = 10
        # Red player (left)
        pygame.draw.circle(self.screen, RED, (radius + 5, 15), radius)
        red_label = self.legend_font.render(self.red_player, True, RED)
        self.screen.blit(red_label, (radius * 2 + 10, 5))
        # Yellow player (right)
        yellow_label = self.legend_font.render(self.yellow_player, True, YELLOW)
        x_yellow_text = WIDTH - yellow_label.get_width() - radius * 2 - 10
        self.screen.blit(yellow_label, (x_yellow_text, 5))
        pygame.draw.circle(self.screen, YELLOW, (WIDTH - radius - 5, 15), radius)

    def draw_board(self, board: np.ndarray):
        self.screen.fill(BLACK)

        for row in range(ROWS):
            for col in range(COLS):

                pygame.draw.rect(
                    self.screen,
                    BLUE,
                    (
                        col * CELL_SIZE,
                        (row + 1) * CELL_SIZE,
                        CELL_SIZE,
                        CELL_SIZE,
                    ),
                )

                piece = board[row][col]

                color = WHITE

                if piece == -1:
                    color = RED
                elif piece == 1:
                    color = YELLOW

                pygame.draw.circle(
                    self.screen,
                    color,
                    (
                        col * CELL_SIZE + CELL_SIZE // 2,
                        (row + 1) * CELL_SIZE + CELL_SIZE // 2,
                    ),
                    CELL_SIZE // 2 - 8,
                )

        pygame.display.update()

    def show_message(self, text: str):
        pygame.draw.rect(self.screen, BLACK, (0, 0, WIDTH, CELL_SIZE))

        self._draw_legend()

        label = self.font.render(text, True, WHITE)
        self.screen.blit(label, (20, 55))

        pygame.display.update()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
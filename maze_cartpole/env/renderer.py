"""Contains the renderer for MazeProjectTemplateRenderer"""
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from maze.core.annotations import override
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.rendering.renderer import Renderer
from maze_cartpole.env.maze_action import CartPoleMazeAction
from maze_cartpole.env.maze_state import CartPoleMazeState


class CartPoleRenderer(Renderer):
    """Matplotlib based rendering class.

    The rendering shows the cart with pole at the current position.
    We use matplotlib here to be compatible with the built-in Maze rendering and debugging tools.

    :param pole_length: The length of the pole to be balanced on the cart.
    :param x_threshold: The threshold to the left and right indicating where the cart is allowed to move.
    """

    def __init__(self, pole_length: float, x_threshold: float):
        self.pole_length = pole_length
        self.x_threshold = x_threshold

    @override(Renderer)
    def render(self, maze_state: CartPoleMazeState, maze_action: Optional[CartPoleMazeAction],
               events: StepEventLog) -> None:
        """Render provided maze_state and maze_action.

        :param maze_state: MazeState to render
        :param maze_action: MazeAction to render
        :param events: Events logged during the step (not used)
        """

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.pole_length)
        cartwidth = 50.0
        cartheight = 30.0

        plt.figure('CartPole', figsize=(8, 4))
        plt.clf()
        plt.xlim([0, screen_width])
        plt.ylim([0, screen_height])
        plt.xticks([])
        plt.yticks([])

        # draw rail
        plt.gca().add_patch(patches.Rectangle((0, carty - 4),
                                              screen_width, 5, facecolor=(0, 0, 0)))

        # draw cart
        cartx = maze_state.cart_position * scale + screen_width / 2.0
        plt.gca().add_patch(patches.Rectangle((cartx, carty + 5), cartwidth, cartheight, facecolor=(0.7, 0.2, 0.2)))
        plt.gca().add_patch(patches.Circle((cartx + cartwidth * 1 / 4, carty + 5), radius=5, facecolor=(0, 0, 0)))
        plt.gca().add_patch(patches.Circle((cartx + cartwidth * 3 / 4, carty + 5), radius=5, facecolor=(0, 0, 0)))

        # draw pole
        plt.gca().add_patch(patches.Rectangle((cartx + cartwidth / 2 - polewidth / 2, carty + cartheight / 2),
                                              polewidth, polelen,
                                              angle=np.rad2deg(-maze_state.pole_angle),
                                              facecolor=(.8, .6, .4)))

        plt.gca().text(20, 20, s=f'step: {events.env_time}')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

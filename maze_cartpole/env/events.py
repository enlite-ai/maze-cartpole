"""Contains custom events for the environment.
Note, that you can have multiple different event classes for different components of your environment."""
from abc import ABC

import numpy as np
from maze.core.log_stats.event_decorators import define_step_stats, define_episode_stats, define_epoch_stats


class CartPoleEvents(ABC):
    """Events related to a specific topic of the Project."""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def cart_moved_away(self):
        """Log if the cart moved out of the  allowed space (left or right)"""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def pole_fell_over(self):
        """Record if the pole fell over."""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    def cart_velocity(self, velocity: float):
        """Record the average cart velocity."""

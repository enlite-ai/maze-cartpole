"""Contains the reward aggregator for the environment."""
from typing import List, Any, Optional

from maze.core.annotations import override
from maze.core.env.maze_state import MazeStateType
from maze.core.env.reward import RewardAggregatorInterface
from maze.utils.bcolors import BColors
from maze_cartpole.env.events import CartPoleEvents


class CartPoleRewardAggregator(RewardAggregatorInterface):
    """Default reward scheme for the environment.
    """

    def __init__(self):
        super().__init__()
        self.steps_beyond_done = None

    @override(RewardAggregatorInterface)
    def get_interfaces(self) -> List[Any]:
        """Specification of the event interfaces this subscriber wants to receive events from.
        Every subscriber must implement this configuration method.
        :return: A list of interface classes"""
        return [CartPoleEvents]

    @override(RewardAggregatorInterface)
    def summarize_reward(self, maze_state: Optional[MazeStateType] = None) -> List[float]:
        """implementation of :class:`~maze.core.env.reward.RewardAggregatorInterface` interface
        """

        terminal_events = list(self.query_events([CartPoleEvents.cart_moved_away, CartPoleEvents.pole_fell_over]))
        done = True if len(terminal_events) > 0 else False

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                BColors.print_colored(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior.", BColors.WARNING
                )
            self.steps_beyond_done += 1
            reward = 0.0

        # in more complex scenarios (e.g., multi-agent or multi-objective) working with lists
        # is often convenient (even though not required for this simple example).
        return [reward]

"""Contains the Action Conversion implementation for the environment."""

from __future__ import annotations

from maze_cartpole.env.maze_action import CartPoleMazeAction
from maze_cartpole.env.maze_state import CartPoleMazeState

from gymnasium import spaces
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionConversionInterface


class DictActionConversion(ActionConversionInterface):
    """Converts agent actions to actual environment maze_actions."""

    @override(ActionConversionInterface)
    def space_to_maze(self, action: dict[str, int], maze_state: CartPoleMazeState) -> CartPoleMazeAction:  # noqa: ARG002
        """Converts agent dictionary action to environment MazeAction object."""
        return CartPoleMazeAction(push_left=action['action'] == 0, push_right=action['action'] == 1)

    @override(ActionConversionInterface)
    def maze_to_space(self, maze_action: CartPoleMazeAction) -> dict[str, int]:
        """Converts environment MazeAction object to agent dictionary action."""
        return {'action': int(maze_action.push_right)}

    @override(ActionConversionInterface)
    def space(self) -> spaces.Dict:
        """Returns Gymnasium dict action space."""
        return spaces.Dict(
            {
                'action': spaces.Discrete(2),  # Move the cart right (1) or move it left (0)
            }
        )

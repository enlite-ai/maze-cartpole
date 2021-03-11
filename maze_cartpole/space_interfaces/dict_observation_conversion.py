"""Contains the Observation Conversion implementation for the environment."""

from typing import Dict

import numpy as np
from gym import spaces

from maze.core.annotations import override
from maze.core.env.observation_conversion import ObservationConversionInterface
from maze_cartpole.env.maze_state import CartPoleMazeState


class DictObservationConversion(ObservationConversionInterface):
    """Environment MazeObservation to dictionary observation.

    :param x_threshold: The threshold of the cart's position.
    :param theta_threshold_radians: The threshold of the pols angle.
    """

    def __init__(self, x_threshold: float, theta_threshold_radians: float):
        self.x_threshold = x_threshold
        self.theta_threshold_radians = theta_threshold_radians

    @override(ObservationConversionInterface)
    def maze_to_space(self, maze_state: CartPoleMazeState) -> Dict[str, np.ndarray]:
        """Converts core environment MazeState to a machine readable agent observation."""

        # Compile dict space observation
        return {'cart_position': np.asarray(maze_state.cart_position, dtype=np.float32),
                'cart_velocity': np.asarray(maze_state.cart_velocity, dtype=np.float32),
                'pole_angle': np.asarray(maze_state.pole_angle, dtype=np.float32),
                'pole_angular_velocity': np.asarray(maze_state.pole_angular_velocity, dtype=np.float32)}

    @override(ObservationConversionInterface)
    def space_to_maze(self, observation: Dict[str, np.ndarray]) -> CartPoleMazeState:
        """Converts agent observation to core environment state (not required for this example)."""
        raise NotImplementedError

    @override(ObservationConversionInterface)
    def space(self) -> spaces.Dict:
        """Return the Gym dict observation space based on the given params.

        :return: Gym space object
            - cart_position: The cart position on the x axis
            - cart_velocity: The cart's movement velocity
            - pole_angle: The angle of the pole on the cart
            - pole_angular_velocity: The angular velocity of the pole on the cart
        """
        return spaces.Dict({
            'cart_position': spaces.Box(low=np.float32(-self.x_threshold * 2), dtype=np.float32,
                                        high=np.float32(self.x_threshold * 2), shape=(1,)),
            'cart_velocity': spaces.Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max,
                                        shape=(1,), dtype=np.float32),
            'pole_angle': spaces.Box(low=np.float32(-self.theta_threshold_radians * 2),
                                     high=np.float32(self.theta_threshold_radians * 2), shape=(1,), dtype=np.float32),
            'pole_angular_velocity': spaces.Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max,
                                                shape=(1,), dtype=np.float32)
        })

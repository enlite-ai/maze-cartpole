"""Contains the MazeEnv implementation. """
from typing import Union

from maze.core.env.action_conversion import ActionConversionInterface
from maze.core.env.core_env import CoreEnv
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationConversionInterface
from maze.core.utils.factory import CollectionOfConfigType, Factory
from maze_cartpole.env.core_env import CartPoleCoreEnvironment


class CartPoleEnvironment(MazeEnv[CartPoleCoreEnvironment]):
    """Maze environment transforming the CoreEnv into a trainable, Gym-style environment.

    :param core_env: The underlying core environment.
    :param action_conversion: An action conversion interface.
    :param observation_conversion: An observation conversion interface.
    """

    def __init__(self,
                 core_env: Union[CoreEnv, dict],
                 action_conversion: CollectionOfConfigType,
                 observation_conversion: CollectionOfConfigType):
        super().__init__(
            core_env=Factory(CartPoleCoreEnvironment).instantiate(core_env),
            action_conversion_dict=Factory(ActionConversionInterface).instantiate_collection(action_conversion),
            observation_conversion_dict=Factory(ObservationConversionInterface).instantiate_collection(
                observation_conversion))

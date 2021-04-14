"""Dummy structured policy for the MazeProjectTemplate."""

from typing import Union, Sequence, Tuple, Optional

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType


class CartPoleDummyHeuristic(Policy):
    """Dummy structured policy for the the Maze Project Template (in it's initial state this is an implementation of
    OpenAI's CartPole-v1). As such the policy simply looks the angle of the pole and moves the cart in the same
    direction in attempt to balance it.

    Useful mainly for showcase of the config scheme and for testing.
    """

    @override(Policy)
    def needs_state(self) -> bool:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface
        """
        return False

    @override(Policy)
    def compute_action(self, observation: ObservationType, maze_state: Optional[MazeStateType] = None,
                       env: Optional[BaseEnv] = None, actor_id: ActorIDType = None, deterministic: bool = False
                       ) -> ActionType:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface
        """
        action = 1 if observation["pole_angle"] > 0 else 0
        return {"action": action}

    @override(Policy)
    def compute_top_action_candidates(self, observation: ObservationType,
                                      num_candidates: int, maze_state: Optional[MazeStateType] = None,
                                      env: Optional[BaseEnv] = None, actor_id: ActorIDType = None,
                                      deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface
        """
        raise NotImplementedError

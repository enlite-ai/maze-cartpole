"""Contains the core env implementation. """
import math
from typing import Union, Tuple, Dict, Any

import numpy as np
from gym.utils import seeding

from maze.core.annotations import override
from maze.core.env.core_env import CoreEnv
from maze.core.env.structured_env import StepKeyType
from maze.core.events.pubsub import Pubsub
from maze.core.utils.factory import Factory
from maze.core.utils.seeding import set_random_states
from maze_cartpole.env.events import CartPoleEvents
from maze_cartpole.env.kpi_calculator import CartPoleKpiCalculator
from maze_cartpole.env.maze_action import CartPoleMazeAction
from maze_cartpole.env.maze_state import CartPoleMazeState
from maze_cartpole.env.renderer import CartPoleRenderer
from maze_cartpole.reward.base_reward import BaseRewardAggregator
from maze_cartpole.reward.default_reward import CartPoleRewardAggregator


class CartPoleCoreEnvironment(CoreEnv):
    """This class holds core structure of the desired environment with the core method 'step'. This function especially
    should encode the behaviour of the env. In this example the OpenAI gym Cartpole-v1 env is implemented for
    the purpose of demonstrating an implementation.

    :param theta_threshold_radians: Angle at which to fail an episode (e.g., 12 * 2 * pi / 360 = 0.20943951).
    :param x_threshold: Position at which to fail an episode (e.g., 2.4).
    :param reward_aggregator: Either an instantiated aggregator or a configuration dictionary.
    """

    def __init__(self, theta_threshold_radians: float, x_threshold: float,
                 reward_aggregator: CartPoleRewardAggregator):
        super().__init__()

        self.theta_threshold_radians = theta_threshold_radians
        self.x_threshold = x_threshold

        # init pubsub for event to reward routing
        self.pubsub = Pubsub(self.context.event_service)

        # KPIs calculation
        self.kpi_calculator = CartPoleKpiCalculator()

        # init reward and register it with pubsub
        self.reward_aggregator = Factory(base_type=BaseRewardAggregator).instantiate(reward_aggregator)
        self.pubsub.register_subscriber(self.reward_aggregator)

        # setup environment
        self.np_random, seed = seeding.np_random()
        self.cart_position = self.np_random.uniform(low=-0.05, high=0.05, size=(1,))[0]
        self.cart_velocity = self.np_random.uniform(low=-0.05, high=0.05, size=(1,))[0]
        self.pole_angle = self.np_random.uniform(low=-0.05, high=0.05, size=(1,))[0]
        self.pole_velocity = self.np_random.uniform(low=-0.05, high=0.05, size=(1,))[0]

        self._setup_env()

        # initialize rendering
        self.renderer = CartPoleRenderer(pole_length=self.length, x_threshold=self.x_threshold)

    def _setup_env(self) -> None:
        """Setup environment."""

        # Setup env here
        self.cart_position = self.np_random.uniform(low=-0.05, high=0.05, size=(1,))[0]
        self.cart_velocity = self.np_random.uniform(low=-0.05, high=0.05, size=(1,))[0]
        self.pole_angle = self.np_random.uniform(low=-0.05, high=0.05, size=(1,))[0]
        self.pole_velocity = self.np_random.uniform(low=-0.05, high=0.05, size=(1,))[0]

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Initialize the events for the env
        self.events = self.pubsub.create_event_topic(CartPoleEvents)
        self.reward_aggregator.steps_beyond_done = None

    @override(CoreEnv)
    def step(self, maze_action: CartPoleMazeAction) \
            -> Tuple[CartPoleMazeState, np.array, bool, Dict[Any, Any]]:
        """Summary of the step (simplified, not necessarily respecting the actual order in the code):
        * Update the cart position and velocity
        * Update the pole position and velocity
        * Update events
        * Calculate reward

        :param maze_action: MazeAction to take.
        :return: state, reward, done, info
        """

        info = {}
        # Implement you step function here and record events

        force = self.force_mag if maze_action.push_right else -self.force_mag
        costheta = math.cos(self.pole_angle)
        sintheta = math.sin(self.pole_angle)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * self.pole_velocity ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole *
                                                                                 costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            self.cart_position = self.cart_position + self.tau * self.cart_velocity
            self.cart_velocity = self.cart_velocity + self.tau * xacc
            self.pole_angle = self.pole_angle + self.tau * self.pole_velocity
            self.pole_velocity = self.pole_velocity + self.tau * thetaacc
        else:  # semi-implicit euler
            self.cart_velocity = self.cart_velocity + self.tau * xacc
            self.cart_position = self.cart_position + self.tau * self.cart_velocity
            self.pole_velocity = self.pole_velocity + self.tau * thetaacc
            self.pole_angle = self.pole_angle + self.tau * self.pole_velocity

        done = False
        if self.cart_position < -self.x_threshold or self.cart_position > self.x_threshold:
            done = True
            self.events.cart_moved_away()
        if self.pole_angle < -self.theta_threshold_radians or self.pole_angle > self.theta_threshold_radians:
            done = True
            self.events.pole_fell_over()

        self.events.cart_velocity(velocity=self.cart_velocity)

        # aggregate reward from events
        reward = self.reward_aggregator.summarize_reward()

        # compile env state
        maze_state = self.get_maze_state()

        return maze_state, reward, done, info

    @override(CoreEnv)
    def get_maze_state(self) -> CartPoleMazeState:
        """Returns the current MazeProjectTemplateMazeState of the environment."""
        return CartPoleMazeState(cart_position=self.cart_position, cart_velocity=self.cart_velocity,
                                 pole_angle=self.pole_angle, pole_angular_velocity=self.pole_velocity)

    @override(CoreEnv)
    def reset(self) -> CartPoleMazeState:
        """Resets the environment to initial state."""
        self._setup_env()
        return self.get_maze_state()

    @override(CoreEnv)
    def close(self) -> None:
        """No additional cleanup necessary."""
        pass

    @override(CoreEnv)
    def seed(self, seed: int) -> None:
        """Seed random state of environment."""
        set_random_states(seed)

    @override(CoreEnv)
    def get_renderer(self) -> CartPoleRenderer:
        """MazeProject renderer module."""
        return self.renderer

    @override(CoreEnv)
    def is_actor_done(self) -> bool:
        """Returns True if the just stepped actor is done, which is different to the done flag of the environment."""
        return False

    @override(CoreEnv)
    def actor_id(self) -> Tuple[StepKeyType, int]:
        """Returns the currently executed actor along with the policy id. The id is unique only with
        respect to the policies (every policy has its own actor 0).
        Note that identities of done actors can not be reused in the same rollout.

        :return: The current actor, as tuple (policy id, actor number).
        """
        return 0, 0

    @property
    @override(CoreEnv)
    def agent_counts_dict(self) -> Dict[Union[str, int], int]:
        """Returns the count of agents for individual sub-steps (or -1 for dynamic agent count).

        As this is a single-step single-agent environment, in which 1 agent gets to act during sub-step 0,
        we return {0: 1}.
        """
        return {0: 1}

    @override(CoreEnv)
    def get_kpi_calculator(self) -> CartPoleKpiCalculator:
        """KPIs are supported."""
        return self.kpi_calculator

    @override(CoreEnv)
    def get_serializable_components(self) -> Dict[str, Any]:
        """List components that should be serialized as part of trajectory data."""
        pass

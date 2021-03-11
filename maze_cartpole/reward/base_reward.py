"""Contains base class for environment specific reward aggregators."""
from abc import abstractmethod
from typing import List, Any

from maze.core.annotations import override
from maze.core.env.reward import RewardAggregatorInterface
from maze.core.events.pubsub import Subscriber
from maze_cartpole.env.events import CartPoleEvents


class BaseRewardAggregator(RewardAggregatorInterface):
    """Event aggregation object dealing with cutting rewards.
    """

    @override(Subscriber)
    def get_interfaces(self) -> List[Any]:
        """Specification of the event interfaces this subscriber wants to receive events from.
        Every subscriber must implement this configuration method.
        :return: A list of interface classes"""
        return [CartPoleEvents]

    @abstractmethod
    def summarize_reward(self) -> List[float]:
        """Assign rewards and penalties according to respective events.
        :return: List of individual event rewards.
        """
        raise NotImplementedError

    @classmethod
    @override(RewardAggregatorInterface)
    def to_scalar_reward(cls, reward: List[float]) -> float:
        """implementation of :class:`~maze.core.env.reward.RewardAggregatorInterface` interface
        """
        return sum(reward)

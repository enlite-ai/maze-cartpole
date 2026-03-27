"""Contains Kpi calculators."""

from __future__ import annotations

from maze_cartpole.env.events import CartPoleEvents

from maze.core.annotations import override
from maze.core.env.maze_state import MazeStateType
from maze.core.log_events.episode_event_log import EpisodeEventLog
from maze.core.log_events.kpi_calculator import KpiCalculator


class CartPoleKpiCalculator(KpiCalculator):
    """Environment specific Key Performance Indicators (KPIs)."""

    @override(KpiCalculator)
    def calculate_kpis(self, episode_event_log: EpisodeEventLog, last_maze_state: MazeStateType) -> dict[str, float]:  # noqa: ARG002
        """Calculates the KPIs at the end of episode."""

        # get overall step count of episode
        step_count = len(episode_event_log.step_event_logs)

        total_velocity = 0
        for event in episode_event_log.query_events(CartPoleEvents.cart_velocity):
            total_velocity += event.velocity

        # compute step normalized velocity of the cart
        return {'average_cart_velocity_per_step': total_velocity / step_count}

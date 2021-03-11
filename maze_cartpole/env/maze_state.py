"""The Project specific Maze State, that is a more detailed (and usually structured) representation of the observation
"""


class CartPoleMazeState:
    """A structured (internal) representation of the observation.

    :param cart_position: The position of the cart. [-4.8, 4.8]
    :param cart_velocity: The velocity of the cart. [-Inf, Inf]
    :param pole_angle: The angle of the pole. [-0.418 rad (-24 deg), 0.418 rad (24 deg)]
    :param pole_angular_velocity: The angular velocity of the pole. [-Inf, Inf]
    """

    def __init__(self, cart_position: float, cart_velocity: float, pole_angle: float, pole_angular_velocity: float):
        self.cart_position = cart_position
        self.cart_velocity = cart_velocity
        self.pole_angle = pole_angle
        self.pole_angular_velocity = pole_angular_velocity

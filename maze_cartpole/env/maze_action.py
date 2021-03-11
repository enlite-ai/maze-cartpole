"""The Project specific Maze Action, that is a more detailed (and usually structured) representation of the action"""


class CartPoleMazeAction:
    """MazeAction object holding the action for the environment.

    :param push_left: Push cart to the left.
    :param push_right: Push cart to the right.
    """

    def __init__(self, push_left: bool, push_right: bool):
        assert push_left or push_right
        assert not (push_left and push_right)
        self.push_left = push_left
        self.push_right = push_right

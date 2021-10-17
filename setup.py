#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages


setup(
    author="enlite.ai",
    author_email='office@enlite.ai',
    name='maze_cartpole',
    version="0.0.2.dev1",
    packages=find_packages(include=['maze_cartpole']),
    include_package_data=True,
    python_requires=">=3.6",
    url='https://github.com/enlite-ai/maze-cartpole',
    install_requires=["maze-rl"],
)

![Banner](https://github.com/enlite-ai/maze/raw/main/docs/source/logos/main_logo.png)

# Maze Project Template 

This repository serves as a template for starting your own projects with Maze.

As such it implements a simple, yet fully functional environment as a placeholder
(a re-implementation of the [Gym CartPole environment](https://gym.openai.com/envs/CartPole-v1/))
for a new Maze project and already contains a hydra config system to train and rollout your agents.

For building your own project we recommend to start with:

1. Renaming the *cartpole_env* specific components to *<your_project_name>*.
2. Implementing the [MazeCoreEnvironment](maze_cartpole/env/core_env.py).
   The class implements all main components of an RL environment such as the step function defining its dynamics.

If this is all very new to you, you can take a look at our 
[docs page](https://maze-rl.readthedocs.io/en/latest/index.html) where the whole framework is explained in more depth. 
Furthermore, it also contains a
[step-by-step getting started guide](https://maze-rl.readthedocs.io/en/latest/getting_started/step_by_step_tutorial.html) 
which iteratively builds a Maze environment entirely from scratch, explains all components in great detail and
links to all relevant documentation pages.

## Project Setup

*  If you not yet installed Maze, please refer to the
[installation instructions](https://maze-rl.readthedocs.io/en/latest/getting_started/installation.html).

 * *Optional*: We also provided an [environment.yml](environment.yml) file
   to create a dedicated conda environment for your project.
   
   Prepare with: `conda env create -f environment.yml`

* Finally, install the project in development mode `pip install -e .` or manually add it to your `PYTHONPATH`.

## Example Commands:
Here are some example commands showing how to train and rollout agents for the exemplary CartPole env.

### Training:

* Train an agent for the env with PPO:

  `maze-run -cn conf_train env=cartpole_env algorithm=ppo`

* Train the env with PPO and a template model:

  `maze-run -cn conf_train env=cartpole_env algorithm=ppo model=cartpole_template_model`

* Train the env with PPO and an environment specific custom model:

  `maze-run -cn conf_train env=cartpole_env algorithm=ppo model=cartpole_custom_model critic=cartpole_custom_state_critic`

* Train the env with PPO and some environment wrappers:

  `maze-run -cn conf_train env=cartpole_env algorithm=ppo wrappers=cartpole_wrappers`

### Rollout:

* Run a rollout with the random policy (default):

  `maze-run -cn conf_rollout env=cartpole_env`

* Run a rollout with the env specific greedy policy:

  `maze-run -cn conf_rollout env=cartpole_env policy=cartpole_heuristic_policy`

* Run a rollout with the greedy policy and render each step:

  `maze-run -cn conf_rollout env=cartpole_env policy=cartpole_heuristic_policy runner=sequential runner.render=True`

* Run and render a rollout with a previously trained policy:

  `maze-run -cn conf_rollout runner=sequential runner.render=True env=cartpole_env policy=torch_policy input_dir=outputs/<exp-dir>/<time-stamp>`

  (Note that we have to use the same overrides for env, model and wrappers as we did during training).

### Experimenting

Following Hydra's experiments configuration workflow
we additionally provide a starting point for convenient
[experimenting with Maze](https://maze-rl.readthedocs.io/en/latest/workflow/experimenting.html) (see `maze_cartpole/conf/experiment`). 

* To start an experiment from a dedicated config, run:

  `maze-run -cn conf_train +experiment=cartpole_hard_ppo`
  
  The overrides in the experiment file will be applied to the defaults specified in
  [conf_train](https://github.com/enlite-ai/maze/blob/main/maze/conf/conf_train.yaml).

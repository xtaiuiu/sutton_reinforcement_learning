##use gym to explore you reinforcement learning algorithms
####1. define your model

you should write a python class and offer three methods, i.e., `step(), render(), reset()`

####2. register in gym

- put the python file into /gym/gym/envs/classic_control/ to use the rendering module
- add the following code to `__init__`.py of that folder:
```python
from gym.envs.classic_control.grid_mdp import GridEnv
```
- than add the following code blocks to the end of the file /gym/gym/envs/`__init__`.py:
```python
    register(
        id='GridWorld-v0',
        entry_point='gym.envs.classic_control:GridEnv',
        max_episode_steps=200,
        reward_threshold=100.0,
        )

```
####3. explore you algorithms on that model:
```python
    env = gym.make('GridWorld-v0')
    env.reset()
    env.render()
```

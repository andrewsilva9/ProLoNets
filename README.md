# ProLoNets

This project houses all of the code for **ProLoNets: Neural-encoding Human Experts' Domain Knowledge to Warm Start Reinforcement Learning**. 

### Requirements

I've gone ahead and made two separate virtualenvs for the OpenAI gym environments and the StarCraft II environments, both of which are built on Python 3.6. In order to work with the SC2 environments, you must have Python >= 3.6, and then installing the requirements in the `sc2_requirements.txt` file should do it. For the gym environments, any Python that works with OpenAI Gym _should_ work, but I haven't tested this.

### Running Experiments

All of the code to run various domains lives in the `runfiles/` directory. 
All file involve a few command line arguments, which I'll review now:

* `-a` or `--agent_type`: Which agent should play through the domain. Details below. Default: `fc`
* `-e` or `--episodes`: How many episodes to run for. Default: 1000
* `-p` or `--processes`: How many concurrent processes to run. Default: 1
* `-s` or `--sl_init`: Should the agent be trained via imitation learning first? Only applies if `agent_type` is `fc`.Default: False
* `-adv` or `--adversary`: Should the agent be mistakenly initialized? Only applies if `agent_type` is `shallow_prolo`. Default: False

For the `-a` or `--agent_type` flag, valid options are:
* `prolo` for a normal ProLoNet agent
* `shallow_prolo` for a non-deepening ProLoNet agent
* `random` for a randomly-initialized ProLoNet agent
* `fc` for a fully-connected agent
* `lstm` for an LSTM agent

#### gym_runner.py

This file runs both of the OpenAI gym domains from the paper, namely cart pole and lunar lander. It has one additional command line argument:
* `-env` or `--env_type`: Which environment to run. Valid options are `cart` and `lunar`. Default: `cart`
In order to run, you  must have the a Python environment with the OpenAI Gym installed. Furthermore, you  must have box2d-py if you want the lunar lander agents to run. The `gym_requirements.txt` file should have everything necessary for a Python 3.6 environment.

Running a ProLoNet agent on lunar lander for 1500 episodes with 4 concurrent processes looks like:
```
python gym_runner.py -a prolo -e 1500 -p 4 -env lunar
```
For the _LOKI_ agent:
```
python gym_runner.py -a fc -e 1500 -p 4 -env lunar -s True
```
And for the _N-Mistake_ agent:
```
python gym_runner.py -a shallow_prolo -e 1500 -p 4 -env lunar -adv True
```

#### minigame_runner.py

This file runs the FindAndDefeatZerglings minigame from the SC2LE. Running this is exactly the same as the `gym_runner.py` runfile, with the exception that no `--env_type` flag exists for this domain. You must also have all of the StarCraft II setup complete, which means having a valid copy of StarCraft II, having Python >= 3.6, and installing the requirements from the `sc2_requirements.txt` file. For information on setting up StarCraft II, refer to [Blizzard's Documentation](https://github.com/Blizzard/s2client-proto) and for the minigame itself, you'll need the map from [DeepMind's repo](https://github.com/deepmind/pysc2).

Running a ProLoNet agent:
```
python minigame_runner.py -a prolo -e 2000 -p 1
```
And a fully-connected agent:
```
python minigame_runner.py -a fc -e 2000 -p 1
```
And an LSTM agent:
```
python minigame_runner.py -a lstm -e 2000 -p 1
```

#### sc_runner.py

This file runs the full SC2 game against in-game AI. In game AI difficulty is set on lines 836-838. Simply changing "Difficult.VeryEasy" to "Difficulty.Easy", "Difficulty.Medium", or "Difficulty.Hard" does the trick. Again, you'll need SC2 and all of the requirements for the appropriate Python environment, as discussed above.
Running a ProLoNet agent:
```
python sc_runner.py -a prolo -e 500 -p 1
```
And a random ProLoNet agent:
```
python sc_runner.py -a random -e 500 -p 1
```
And a non-deepening ProLoNet agent:
```
python sc_runner.py -a shallow_prolo -e 500 -p 1
```


### Questions:
Any questions about the work or the code can be either opened up as GitHub issues, or sent directly to me at andrew.silva@gatech.edu

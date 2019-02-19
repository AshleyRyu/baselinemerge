"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent. 
"""

from design_agent_and_env import design_agent_and_env
# from options import parse_options
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env, parse_options
from agent import Agent
from run_HAC import run_HAC


## FLAG를 디폴트로서 여기서 만들어준다. 높.
# parser = argparse.ArgumentParser()
# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
FLAGS = parse_options()
# FLAGS = {}

# Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file. 
agent, env = design_agent_and_env(FLAGS)

# Begin training
run_HAC(FLAGS,env,agent)

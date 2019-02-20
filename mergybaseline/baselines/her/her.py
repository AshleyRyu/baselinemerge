import os
import sys
sys.path.insert(0, 'baselines/her/experiment')
sys.path.insert(0, 'baselines/her')

import click
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds, tf_util
from baselines.common.mpi_moments import mpi_moments
# import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.common.cmd_util import parse_options

from baselines.her.design_agent_and_env import design_agent_and_env ##

import config as config 
# import configure_ddpg, configure_dims 

import gym #jw

def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]


# def train(*, env_name, policy, rollout_worker, evaluator,
def train(*, env_name, policy, agent, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, demo_file, FLAGS, **kwargs):
        #   save_path, demo_file, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()
    
    if save_path:
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')

    logger.info("Debug @ basemerge : Training...") # 마지막 ### 8
    best_success_rate = -1

    if policy.bc_loss == 1: policy.init_demo_buffer(demo_file) #initialize demo buffer if training with demonstrations

    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts(FLAGS)
            # episode = rollout_worker.generate_rollouts(FLAGS) ##
            # episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
                # env.render() #jw


            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts(FLAGS)

        # record logs 여기서 찍는다
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        # for key, val in rollout_worker.logs('train'): ##original
        #     logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train1'): ##A.R
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train2'): ##A.R
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_path:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return policy


def learn(*, network, env, total_timesteps, ### 4
    seed=None,
    eval_env=None,
    replay_strategy='future',
    policy_save_interval=5,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_path=None,
    save_path=None,
    **kwargs
):
    
    print("-------------------JW Debug learn func @ her.py with hrl baseline merge ----------------------")
    override_params = override_params or {}
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        num_cpu = MPI.COMM_WORLD.Get_size()

    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    env_name = env.specs[0].id
    params['env_name'] = env_name
    # print(env_name)

    
    params['replay_strategy'] = replay_strategy
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
         json.dump(params, f)
    params = config.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs

    if demo_file is not None:
        params['bc_loss'] = 1
    params.update(kwargs)

    config.log_params(params, logger=logger) ### 5

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************') ### 6
        logger.warn()


    dims = config.configure_dims(params)
    # policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return, FLAGS=FLAGS, agent_params=agent_params)
    #===============================#
    FLAGS = parse_options() ## Prepare params for HAC.
    
    FLAGS.layers = 2    # Enter number of levels in agent hierarchy

    FLAGS.time_scale = 10    # Enter max sequence length in which each policy will specialize

    # Enter max number of atomic actions.  
    # This will typically be FLAGS.time_scale**(FLAGS.layers).  
    # However, in the UR5 Reacher task, we use a shorter episode length.
    # max_actions = FLAGS.time_scale**(FLAGS.layers-1)*6 
    max_actions = 1000    

    timesteps_per_action = 15    # Provide the number of time steps per atomic action.

    agent_params = {}

    # Define percentage of actions that a subgoal level (i.e. level i > 0) will test subgoal actions
    agent_params["subgoal_test_perc"] = 0.3

    # Define subgoal penalty for missing subgoal.  Please note that by default the Q value target for missed subgoals does not include Q-value of next state (i.e, discount rate = 0).  As a result, the Q-value target for missed subgoal just equals penalty.  For instance in this 3-level UR5 implementation, if a level proposes a subgoal and misses it, the Q target value for this action would be -10.  To incorporate the next state in the penalty, go to the "penalize_subgoal" method in the "layer.py" file.
    agent_params["subgoal_penalty"] = -FLAGS.time_scale     

    # Define exploration noise that is added to both subgoal actions and atomic actions.  Noise added is Gaussian N(0, noise_percentage * action_dim_range)    
    agent_params["atomic_noise"] = [0.1 for i in range(3)]
    agent_params["subgoal_noise"] = [0.03 for i in range(6)]

    # Define number of episodes of transitions to be stored by each level of the hierarchy
    agent_params["episodes_to_store"] = 500

    # Provide training schedule for agent.  
    # Training by default will alternate between exploration and testing.  
    # Hyperparameter below indicates number of exploration episodes.  
    # Testing occurs for 100 episodes.  To change number of testing episodes, go to "ran_HAC.py". 
    agent_params["num_exploration_episodes"] = 50
    # policy = config.configure_ddpg(params, FLAGS, dims, reuse, use_mpi, clip_return) # 이걸 어떻게 해야해!
    
# def configure_ddpg(dims, params, FLAGS, agent_params, reuse=False, use_mpi=True, clip_return=True):
    policy = config.configure_ddpg(dims=dims, params=params, FLAGS=FLAGS, agent_params=agent_params, reuse=False, use_mpi=True, clip_return=True) # 이걸 어떻게 해야해!
    # 원래 dims, params, reuse=False, use_mpi=True, clip_return=True

    # agent = design_agent_and_env(FLAGS, env, dims=dims, params=params, clip_return=clip_return) ## make agent(TD3) for HAC.
    # policy = design_agent_and_env(FLAGS, env, dims=dims, params=params, clip_return=clip_return) ## make agent(TD3) for HAC.
    #===============================#
    if load_path is not None:
        tf_util.load_variables(load_path)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
        ############hrl################

        ###############################
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env = eval_env or env

    ## Done with prepare
    # run_HAC(FLAGS, agent)
    # agent = design_agent_and_env(FLAGS, env, dims=dims, params=params, clip_return=clip_return) ## 원래거
    agent = design_agent_and_env(FLAGS, env, dims, policy, logger, rollout_params, eval_params,agent_params, monitor=True)

    # rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    # ##
    # # rollout_worker_high = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    # ##
    # evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params) ## 뭐하는 놈임

    n_cycles = params['n_cycles']
    n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size
    # print("#######################################n_epoch = {}".format(n_epochs)) ### 7

    return train(
        save_path=save_path, 
        env_name=env_name, #jw
        policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, demo_file=demo_file, FLAGS=FLAGS)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--total_timesteps', type=int, default=int(5e5), help='the number of timesteps to run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default = 'PATH/TO/DEMO/DATA/FILE.npz', help='demo data file path')
def main(**kwargs):
    learn(**kwargs)


if __name__ == '__main__':
    main()

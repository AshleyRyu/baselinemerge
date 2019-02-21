# from collections import deque

# import numpy as np
# import pickle

# from baselines.her.util import convert_episode_to_batch_major, store_args


# class RolloutWorker:

#     @store_args
#     def __init__(self, venv, policy, dims, logger, T, rollout_batch_size=1,
#                  exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
#                  random_eps=0, history_len=100, render=False, monitor=False, **kwargs):
#         """Rollout worker generates experience by interacting with one or many environments.

#         Args:
#             make_env (function): a factory function that creates a new instance of the environment
#                 when called
#             policy (object): the policy that is used to act
#             dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
#             logger (object): the logger that is used by the rollout worker
#             rollout_batch_size (int): the number of parallel rollouts that should be used
#             exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
#                 current policy without any exploration
#             use_target_net (boolean): whether or not to use the target net for rollouts
#             compute_Q (boolean): whether or not to compute the Q values alongside the actions
#             noise_eps (float): scale of the additive Gaussian noise
#             random_eps (float): probability of selecting a completely random action
#             history_len (int): length of history for statistics smoothing
#             render (boolean): whether or not to render the rollouts
#         """

#         assert self.T > 0

#         self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

#         self.success_history = deque(maxlen=history_len)
#         self.Q_history = deque(maxlen=history_len)

#         self.n_episodes = 0
#         self.reset_all_rollouts()
#         self.clear_history()

#         ############################################ hrl multi agent ###################################################
#         self.initial_high_goal_gt_tilda = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
#         self.initial_high_goal_gt = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
#         self.high_level_train_step = 10
#         self.discount = 0.99
#         self.total_timestep = 0
#         ################################################################################################################
#     ##
#     def reset_rollout(self, i):
#         """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
#         and `g` arrays accordingly.
#         """
#         obs = self.envs[i].reset()
#         self.initial_o[i] = obs['observation']
#         self.initial_ag[i] = obs['achieved_goal']
#         self.g[i] = obs['desired_goal']
#     ##    
#     def reset_all_rollouts(self):
#         self.obs_dict = self.venv.reset()
#         self.initial_o = self.obs_dict['observation']
#         self.initial_ag = self.obs_dict['achieved_goal']
#         self.g = self.obs_dict['desired_goal']

#     def generate_rollouts(self):
#         """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
#         policy acting on it accordingly.
#         """
#         self.reset_all_rollouts()

#         # compute observations
#         o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
#         ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
#         o[:] = self.initial_o
#         ag[:] = self.initial_ag

#         # generate episodes
#         obs, achieved_goals, acts, goals, successes = [], [], [], [], []
#         dones = []
#         info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
#         Qs = []

#         ####################### hrl #############################

#         Rt_high_sum = np.zeros((self.rollout_batch_size, 1), np.float32)
#         total_timestep = 1
#         high_goal_gt = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
#         #high_goal_gt_tilda = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
#         high_old_obj_st = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)

#         u_temp = np.empty((self.rollout_batch_size, self.dims['u']), np.float32)

#         low_nn_at = np.zeros((self.high_level_train_step*self.rollout_batch_size, self.dims['u']),
#                                   np.float32).reshape(self.rollout_batch_size, self.high_level_train_step, self.dims['u'])
#         low_nn_st = np.zeros((self.high_level_train_step*self.rollout_batch_size, self.dims['o']),
#                                   np.float32).reshape(self.rollout_batch_size, self.high_level_train_step, self.dims['o'])
#         intrinsic_reward = np.zeros((self.rollout_batch_size, 1), np.float32)

#         high_goal_gt[:] = self.initial_high_goal_gt
#         #high_goal_gt_tilda[:] = self.initial_high_goal_gt_tilda

#         ##########################################################

#         for t in range(self.T):
#             policy_output = self.policy.get_actions(
#                 o, ag, self.g,
#                 compute_Q=self.compute_Q,
#                 noise_eps=self.noise_eps if not self.exploit else 0.,
#                 random_eps=self.random_eps if not self.exploit else 0.,
#                 use_target_net=self.use_target_net)

#             if self.compute_Q:
#                 u, Q = policy_output
#                 Qs.append(Q)
#             else:
#                 u = policy_output

#             if u.ndim == 1:
#                 # The non-batched case should still have a reasonable shape.
#                 u = u.reshape(1, -1)
            
#             try:

#             o_new = np.empty((self.rollout_batch_size, self.dims['o']))
#             ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
#             success = np.zeros(self.rollout_batch_size)
#             # compute new states and observations
#             obs_dict_new, _, done, info = self.venv.step(u)
#             o_new = obs_dict_new['observation']
#             ag_new = obs_dict_new['achieved_goal']
#             success = np.array([i.get('is_success', 0.0) for i in info])
#             # o_new = np.empty((self.rollout_batch_size, self.dims['o']))
#             # ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
#             # success = np.zeros(self.rollout_batch_size)
#             # # compute new states and observations
#             # obs_dict_new, _, done, info = self.venv.step(u)
#             # o_new = obs_dict_new['observation']
#             # ag_new = obs_dict_new['achieved_goal']
#             # success = np.array([i.get('is_success', 0.0) for i in info])

#             if any(done):
#                 # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
#                 # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
#                 # after a reset
#                 break

#             for i, info_dict in enumerate(info): # HER를 위해 indexing 하는 부분, 재윤님 코드엔 없다
#                 for idx, key in enumerate(self.info_keys):
#                     info_values[idx][t, i] = info[i][key]

#             if np.isnan(o_new).any():
#                 self.logger.warn('NaN caught during rollout generation. Trying again...')
#                 self.reset_all_rollouts()
#                 return self.generate_rollouts()

#             dones.append(done)
#             obs.append(o.copy())
#             # print("############## obs = {}".format(obs))
#             achieved_goals.append(ag.copy())
#             successes.append(success.copy())
#             acts.append(u.copy())
#             goals.append(self.g.copy())
#             o[...] = o_new
#             ag[...] = ag_new
#         obs.append(o.copy())
#         achieved_goals.append(ag.copy())

#         episode = dict(o=obs,
#                        u=acts,
#                        g=goals,
#                        ag=achieved_goals)
#         for key, value in zip(self.info_keys, info_values):
#             episode['info_{}'.format(key)] = value

#         # stats
#         successful = np.array(successes)[-1, :]
#         assert successful.shape == (self.rollout_batch_size,)
#         success_rate = np.mean(successful)
#         self.success_history.append(success_rate)
#         if self.compute_Q:
#             self.Q_history.append(np.mean(Qs))
#         self.n_episodes += self.rollout_batch_size

#         return convert_episode_to_batch_major(episode)

#     def clear_history(self):
#         """Clears all histories that are used for statistics
#         """
#         self.success_history.clear()
#         self.Q_history.clear()

#     def current_success_rate(self):
#         return np.mean(self.success_history)

#     def current_mean_Q(self):
#         return np.mean(self.Q_history)

#     def save_policy(self, path):
#         """Pickles the current policy for later inspection.
#         """
#         with open(path, 'wb') as f:
#             pickle.dump(self.policy, f)

#     def logs(self, prefix='worker'):
#         """Generates a dictionary that contains all collected statistics.
#         """
#         logs = []
#         logs += [('success_rate', np.mean(self.success_history))]
#         if self.compute_Q:
#             logs += [('mean_Q', np.mean(self.Q_history))]
#         logs += [('episode', self.n_episodes)]

#         if prefix is not '' and not prefix.endswith('/'):
#             return [(prefix + '/' + key, val) for key, val in logs]
#         else:
#             return logs

from collections import deque

import numpy as np
import pickle

from baselines.her.util import convert_episode_to_batch_major, store_args

## from run_HAC.py
TEST_FREQ = 2
num_test_episodes = 100

class RolloutWorker:

    @store_args
    def __init__(self, venv, policy, dims, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, monitor=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker -> Deleted
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """

        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()
        
        ############################################ hrl multi agent ###################################################
        self.initial_high_goal_gt_tilda = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_high_goal_gt = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        self.high_level_train_step = 10
        self.discount = 0.99
        self.total_timestep = 0
        ################################################################################################################
        # ############################################ hrl multi agent ###################################################
        # self.initial_high_goal_gt_tilda = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        # self.initial_high_goal_gt = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        # self.high_level_train_step = 10
        # self.discount = 0.99
        # self.total_timestep = 0
        # ################################################################################################################

    def reset_all_rollouts(self):
        self.obs_dict = self.venv.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal']

    def generate_rollouts(self, FLAGS):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        print("마침내 generate_rollout!")

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        dones = []
        info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []

        ####################### hrl #############################

        # Rt_high_sum = np.zeros((self.rollout_batch_size, 1), np.float32)
        # total_timestep = 1
        # high_goal_gt = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        # #high_goal_gt_tilda = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        # high_old_obj_st = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)

        # u_temp = np.empty((self.rollout_batch_size, self.dims['u']), np.float32)

        # low_nn_at = np.zeros((self.high_level_train_step*self.rollout_batch_size, self.dims['u']),
        #                           np.float32).reshape(self.rollout_batch_size, self.high_level_train_step, self.dims['u'])
        # low_nn_st = np.zeros((self.high_level_train_step*self.rollout_batch_size, self.dims['o']),
        #                           np.float32).reshape(self.rollout_batch_size, self.high_level_train_step, self.dims['o'])
        # intrinsic_reward = np.zeros((self.rollout_batch_size, 1), np.float32)

        # high_goal_gt[:] = self.initial_high_goal_gt
        # #high_goal_gt_tilda[:] = self.initial_high_goal_gt_tilda

        ##########################################################

        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)
                # FLAGS=FLAGS)
            

            # policy_output = self.policy.get_actions(
            #     o, ag, self.g,
            #     compute_Q=self.compute_Q,
            #     noise_eps=self.noise_eps if not self.exploit else 0.,
            #     random_eps=self.random_eps if not self.exploit else 0.,
            #     use_target_net=self.use_target_net)

            ## from run_HAC.py
            # Determine training mode.  If not testing and not solely training, interleave training and testing to track progress
            # mix_train_test = False
            # if not FLAGS.test and not FLAGS.train_only:
            #     mix_train_test = True

            ## from run_HAC.py, 이 뒤로 다 indentation해줌
            # Evaluate policy every TEST_FREQ batches if interleaving training and testing
            # if mix_train_test and t % TEST_FREQ == 0:
            #     print("\n--- HAC TESTING ---")
            #     # agent.FLAGS.test = True ## agent를 인스턴스로 받아야하나 ㅡㅡ
            #     num_episodes = num_test_episodes            

            #     # Reset successful episode counter
            #     successful_episodes = 0

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # print("Rollout. o_new={}, ag_new={},success={}".format(o_new,ag_new,success))
            # compute new states and observations
            obs_dict_new, _, done, info = self.venv.step(u)
            # print("HERE")
            # print("#########Debug##########")
            o_new = obs_dict_new['observation']
            # print("observation high : {}".format(o_new))
            ag_new = obs_dict_new['achieved_goal']
            success = np.array([i.get('is_success', 0.0) for i in info])

            if any(done):
                # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
                # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
                # after a reset
                # 여기에서 우리는 모든 환경이 동일한 단계 수라고 가정합니다. 
                # 그래서 envs가 vecenvs를 사용하여 수행 한 트릭을 반환 할 때마다 롤아웃을 종료합니다.
                # 왜냐하면 그것들은 이미 재설정된 후의 관찰이기 때문이다.
                
                break

            for i, info_dict in enumerate(info):
                for idx, key in enumerate(self.info_keys):
                    info_values[idx][t, i] = info[i][key]

            if np.isnan(o_new).any():
                # self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            dones.append(done)
            obs.append(o.copy())
            # print("############## obs = {}".format(obs))
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

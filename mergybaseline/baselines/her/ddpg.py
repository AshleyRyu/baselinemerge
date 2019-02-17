from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, convert_episode_to_batch_major)
from baselines.her.normalizer import Normalizer
from baselines.her.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.common import tf_util


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


global DEMO_BUFFER #buffer for demonstrations

class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 bc_loss, q_filter, num_demo, demo_batch_size, prm_loss_weight, aux_loss_weight,
                 
                #  sample_transitions, gamma, reuse=False, **kwargs):
                 sample_transitions, gamma, td3_policy_freq, td3_policy_noise, td3_noise_clip,  reuse=False, **kwargs): ##
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).
            Added functionality to use demonstrations for training to Overcome exploration problem.

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
            bc_loss: whether or not the behavior cloning loss should be used as an auxilliary loss
            q_filter: whether or not a filter on the q value update should be used when training with demonstartions
            num_demo: Number of episodes in to be used in the demonstration buffer
            demo_batch_size: number of samples to be used from the demonstrations buffer, per mpi thread
            prm_loss_weight: Weight corresponding to the primary loss
            aux_loss_weight: Weight corresponding to the auxilliary loss also called the cloning loss
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        # self.dimo1= self.input_dims['o1'] ##A.R add for TD3 (has obs0, obs1)
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']
        #추가된 내용
        #parameters for using TD3 variant of DDPG
        #https://arxiv.org/abs/1802.09477
        self.td3_policy_freq = td3_policy_freq
        self.td3_policy_noise = td3_policy_noise
        self.td3_noise_clip = td3_noise_clip

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
        # for key in ['o', 'o1', 'g']: #o1 added by A.R
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key]) 
        # origin : buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key]) 
        # buffer_shapes = {key: (self.T-1 if key != 'o' and key != 'o1' else self.T, *input_shapes[key]) #A.Rㅇ
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

        global DEMO_BUFFER
        DEMO_BUFFER = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions) #initialize the demo buffer; in the same way as the primary data buffer

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
    # def _preprocess_og(self, o, o1, ag, g): #A.R
        if self.relative_goals: ## goal reshape 해주는 곳. ag vs g..흠
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag) #상대적인 골로 만들어 주는구나?..
            '''
            def simple_goal_subtract(a, b):
            assert a.shape == b.shape
            return a - b
            '''
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        # o1 = np.clip(o1, -self.clip_obs, self.clip_obs) #A.R
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        # return o, o1, g
        return o, g

    def step(self, obs):
        actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'])
        # print("for debug, obs : {}".format(obs['observation']))
        return actions, None, None, None


    # def get_actions(self, o, o1, ag, g, noise_eps=0., random_eps=0., use_target_net=False, ##o1이 target 네트워크
    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        # o, o1, g = self._preprocess_og(o, o1, ag, g) ##
        o, g = self._preprocess_og(o, ag, g) 
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            # policy.o1_tf: o1.reshape(-1, self.dimo), ##
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def init_demo_buffer(self, demoDataFile, update_stats=True): #function that initializes the demo buffer

        demoData = np.load(demoDataFile) #load the demonstration data from data file
        info_keys = [key.replace('info_', '') for key in self.input_dims.keys() if key.startswith('info_')]
        info_values = [np.empty((self.T - 1, 1, self.input_dims['info_' + key]), np.float32) for key in info_keys]

        demo_data_obs = demoData['obs']
        demo_data_acs = demoData['acs']
        demo_data_info = demoData['info']

        for epsd in range(self.num_demo): # we initialize the whole demo buffer at the start of the training
            obs, acts, goals, achieved_goals = [], [] ,[] ,[]
            i = 0
            for transition in range(self.T - 1):
                obs.append([demo_data_obs[epsd][transition].get('observation')])
                acts.append([demo_data_acs[epsd][transition]])
                goals.append([demo_data_obs[epsd][transition].get('desired_goal')])
                achieved_goals.append([demo_data_obs[epsd][transition].get('achieved_goal')])
                for idx, key in enumerate(info_keys):
                    info_values[idx][transition, i] = demo_data_info[epsd][transition][key]


            obs.append([demo_data_obs[epsd][self.T - 1].get('observation')])
            achieved_goals.append([demo_data_obs[epsd][self.T - 1].get('achieved_goal')])

            episode = dict(o=obs,
                           u=acts,
                           g=goals,
                           ag=achieved_goals)
            for key, value in zip(info_keys, info_values):
                episode['info_{}'.format(key)] = value

            episode = convert_episode_to_batch_major(episode)
            global DEMO_BUFFER
            DEMO_BUFFER.store_episode(episode) # create the observation dict and append them into the demonstration buffer
            logger.debug("Demo buffer size currently ", DEMO_BUFFER.get_current_size()) #print out the demonstration buffer size

            if update_stats:
                # add transitions to normalizer to normalize the demo data as well
                episode['o_2'] = episode['o'][:, 1:, :]
                episode['ag_2'] = episode['ag'][:, 1:, :]
                num_normalizing_transitions = transitions_in_episode_batch(episode)
                transitions = self.sample_transitions(episode, num_normalizing_transitions)

                o, g, ag = transitions['o'], transitions['g'], transitions['ag']
                transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
                # No need to preprocess the o_2 and g_2 since this is only used for stats

                self.o_stats.update(transitions['o'])
                self.g_stats.update(transitions['g'])

                self.o_stats.recompute_stats()
                self.g_stats.recompute_stats()
            episode.clear()

        logger.info("Demo buffer size: ", DEMO_BUFFER.get_current_size()) #print out the demonstration buffer size

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self):
        if self.bc_loss: #use demonstration buffer to sample as well if bc_loss flag is set TRUE
            transitions = self.buffer.sample(self.batch_size - self.demo_batch_size)
            global DEMO_BUFFER
            transitions_demo = DEMO_BUFFER.sample(self.demo_batch_size) #sample from the demo buffer
            for k, values in transitions_demo.items():
                rolloutV = transitions[k].tolist()
                for v in values:
                    rolloutV.append(v.tolist())
                transitions[k] = np.array(rolloutV)
        else:
            transitions = self.buffer.sample(self.batch_size) #otherwise only sample from primary buffer

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        # o1, o1_2, g = transitions['o1'], transitions['o1_2'] ## A.R
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target1_net_op)
        self.sess.run(self.init_target2_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)
        self.sess.run(self.update_target1_net_op)
        self.sess.run(self.update_target2_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0 #######################이게 왜걸리지? 왜 다시 안걸리지?
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        # print("DEBUG, {}".format(res))
        
        return res

    def _create_network(self, reuse=False): ## num_demo 추가
        logger.info("Debug : Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()
        # self.num_demo = num_demo

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        #choose only the demo buffer samples
        mask = np.concatenate((np.zeros(self.batch_size - self.demo_batch_size), np.ones(self.demo_batch_size)), axis = 0)

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
            print("tf.variable_scope(main) = {}".format(tf.variable_scope('target1')))

        with tf.variable_scope('target1') as vs:
            if reuse:
                vs.reuse_variables()
            target1_batch_tf = batch_tf.copy()
            target1_batch_tf['o'] = batch_tf['o_2']
            target1_batch_tf['g'] = batch_tf['g_2']
            self.target1 = self.create_actor_critic(
                target1_batch_tf, net_type='target1', **self.__dict__)
            vs.reuse_variables()
            print("tf.variable_scope(target1) = {}".format(tf.variable_scope('target1')))
            # print("batch= {}".format(target1_batch_tf))
            # print(type('target')) #<class 'baselines.her.actor_critic.ActorCritic'>
        assert len(self._vars("main")) == len(self._vars("target1"))

        with tf.variable_scope('target2') as vs:
            if reuse:
                vs.reuse_variables()
            target2_batch_tf = batch_tf.copy()
            target2_batch_tf['o'] = batch_tf['o_2']
            target2_batch_tf['g'] = batch_tf['g_2']
            self.target2 = self.create_actor_critic(
                target2_batch_tf, net_type='target2', **self.__dict__)
            vs.reuse_variables()
            print("tf.variable_scope(target2) = {}".format(tf.variable_scope('target2')))
            print("batch= {}".format(target2_batch_tf))
        assert len(self._vars("main")) == len(self._vars("target2"))

        for nd in range(self.num_demo):       

            ##A.R
            ##Compute the target Q value, Q1과 Q2중에 min값을 사용한다.

            # target1_Q_pi_tf = self.target1.Q_pi_tf ##A.R policy training
            # target2_Q_pi_tf = self.target2.Q_pi_tf ##A.R
            # target_Q_pi_tf = tf.minimum(target1_Q_pi_tf, target2_Q_pi_tf)
            target1_Q_tf = self.target1.Q_tf ##A.R policy training
            target2_Q_tf = self.target2.Q_tf ##A.R
            # print('target1={}/////target2={}'.format(target1_Q_tf,target2_Q_tf))
            target_Q_tf = tf.minimum(target1_Q_tf, target2_Q_tf)
            # print("{}///{}///{}".format(target1_Q_pi_tf,target2_Q_pi_tf,tf.minimum(target1_Q_pi_tf, target2_Q_pi_tf)))
            ####
            #TD3에서 빠진 코드 :target_Q = reward + (done * discount * target_Q).detach()(L109) ->L428에서 해주고 clip한다

            # loss functions
            # for policy training, Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # target_Q_pi_tf = self.target.Q_pi_tf #original code
            clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
            target_Q_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_tf, *clip_range)
            # self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))
            ##
            # current_Q1, current_Q2 = self.critic(state, action)

            # for critic training, Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
            # target_Q_pi_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_tf, *clip_range) #original code
            
            # self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf)) #critic taining 
            
            ## Get current Q estimates, for critic Q
            current_Q1 = self.main.Q_tf ##A.R
            current_Q2 = self.main.Q_tf
            # print("Q1={}".format(current_Q1))

            ## Compute critic loss
            ## Torch => critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 
            self.Q_loss_tf = tf.losses.mean_squared_error(current_Q1, target_Q_tf)+ tf.losses.mean_squared_error(current_Q2,target_Q_tf)
            # self.Q_loss_tf = tf.losses.mean_squared_error(current_Q1, target_Q_tf)+ tf.losses.mean_squared_error(current_Q2,target_Q_tf)
            # print("critic_loss ={}".format(self.Q_loss_tf))

            Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
            assert len(self._vars('main/Q')) == len(Q_grads_tf)

            ## Optimize the critic 아담 옵티마이저
            self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
            assert len(self._vars('main/Q')) == len(Q_grads_tf)
            self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
            self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))

            # ## Delayed policy updates
            if nd % self.td3_policy_freq == 0:
                # print("num_demo = {}".format(nd))
                target1_Q_pi_tf = self.target1.Q_pi_tf ##A.R policy training
                target2_Q_pi_tf = self.target2.Q_pi_tf ##A.R
                tf.print(target1_Q_pi_tf, [target1_Q_pi_tf])
                tf.print(target2_Q_pi_tf, [target2_Q_pi_tf])
                # print(target2_Q_pi_tf)
                target_Q_pi_tf = tf.minimum(target1_Q_pi_tf, target2_Q_pi_tf)

                # target_Q_pi_tf = self.target.Q_pi_tf
                clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
                target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
                self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))
                # Compute actor loss
                if self.bc_loss ==1 and self.q_filter == 1 : # train with demonstrations and use bc_loss and q_filter both
                    maskMain = tf.reshape(tf.boolean_mask(self.main.Q_tf > self.main.Q_pi_tf, mask), [-1]) #where is the demonstrator action better than actor action according to the critic? choose those samples only
                    #define the cloning loss on the actor's actions only on the samples which adhere to the above masks
                    self.cloning_loss_tf = tf.reduce_sum(tf.square(tf.boolean_mask(tf.boolean_mask((self.main.pi_tf), mask), maskMain, axis=0) - tf.boolean_mask(tf.boolean_mask((batch_tf['u']), mask), maskMain, axis=0)))
                    self.pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main.Q_pi_tf) #primary loss scaled by it's respective weight prm_loss_weight
                    self.pi_loss_tf += self.prm_loss_weight * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u)) #L2 loss on action values scaled by the same weight prm_loss_weight
                    self.pi_loss_tf += self.aux_loss_weight * self.cloning_loss_tf #adding the cloning loss to the actor loss as an auxilliary loss scaled by its weight aux_loss_weight

                elif self.bc_loss == 1 and self.q_filter == 0: # train with demonstrations without q_filter
                    self.cloning_loss_tf = tf.reduce_sum(tf.square(tf.boolean_mask((self.main.pi_tf), mask) - tf.boolean_mask((batch_tf['u']), mask)))
                    self.pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main.Q_pi_tf)
                    self.pi_loss_tf += self.prm_loss_weight * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                    self.pi_loss_tf += self.aux_loss_weight * self.cloning_loss_tf

                else: #If  not training with demonstrations
                    self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
                    self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                # self.pi_loss_tf = -tf.reduce_mean(self.main.pi_tf) ## what about target1?
                # self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                # actor_loss = -tf.reduce_mean(self.main.Q_tf)
                # actor_loss += self.action_l2 * tf.reduce_mean(tf.square(self.main.Q_tf / self.max_u))

                pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
                assert len(self._vars('main/pi')) == len(pi_grads_tf)

                # Optimize the actor 
                # Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
                self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)
                assert len(self._vars('main/pi')) == len(pi_grads_tf)
                self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
                self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

                # Update the frozen target models
            ## torch code
                # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                

                self.main_vars = self._vars('main/Q') + self._vars('main/pi')
                self.target1_vars = self._vars('target1/Q') + self._vars('target1/pi') ##A.R
                self.target2_vars = self._vars('target2/Q') + self._vars('target2/pi') ##A.R
                if target_Q_pi_tf == target1_Q_pi_tf:
                    target_vars = self.target1_vars
                else:
                    target_vars = self.target2_vars
                # self.target_vars = self._vars('target/Q') + self._vars('target/pi') #original
                self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
                self.init_target1_net_op = list(
                    map(lambda v: v[0].assign(v[1]), zip(self.target1_vars, self.main_vars)))
                self.init_target2_net_op = list(
                    map(lambda v: v[0].assign(v[1]), zip(self.target2_vars, self.main_vars)))

                self.update_target_net_op = list(
                    map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(target_vars, self.main_vars)))
                self.update_target1_net_op = list(
                    map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(target_vars, self.main_vars)))
                self.update_target2_net_op = list(
                    map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(target_vars, self.main_vars)))


                tf.variables_initializer(self._global_vars('')).run()
                self._sync_optimizers()
                self._init_target_net()



        # Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        # pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        # assert len(self._vars('main/Q')) == len(Q_grads_tf)
        # assert len(self._vars('main/pi')) == len(pi_grads_tf)
        # self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        # self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        # self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        # self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        # self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        # self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        # self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        # self.target1_vars = self._vars('target1/Q') + self._vars('target1/pi') ##A.R
        # self.target2_vars = self._vars('target2/Q') + self._vars('target2/pi') ##A.R
        # # self.target_vars = self._vars('target/Q') + self._vars('target/pi') #original
        # self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        # self.init_target1_net_op = list(
        #     map(lambda v: v[0].assign(v[1]), zip(self.target1_vars, self.main_vars)))
        # self.init_target2_net_op = list(
        #     map(lambda v: v[0].assign(v[1]), zip(self.target2_vars, self.main_vars)))


            
        # self.update_target_net_op = list(
        #     map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        #original
        # self.init_target_net_op = list(
        #     map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        # self.update_target_net_op = list(
        #     map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # # initialize all variables
        # tf.variables_initializer(self._global_vars('')).run()
        # self._sync_optimizers()
        # self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                            #  'main', 'target', 'lock', 'env', 'sample_transitions', #original code
                            'main', 'target1', 'target2', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

    def save(self, save_path):
        tf_util.save_variables(save_path)


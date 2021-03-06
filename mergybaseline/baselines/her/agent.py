import numpy as np
# from environment import Environment
import pickle as cpickle
import tensorflow as tf
import os
import pickle as cpickle
##
# import baselines.her.experiment.config as config
# import baselines.her.config as config
# from baselines.her.config import configure_ddpg
# from .experiment.config import config
import sys
sys.path.insert(0, 'baselines/her/experiment')
import config

sys.path.insert(0, 'baselines/her')
from layer import Layer


# import .experiment.config as config
#
# Below class instantiates an agent
class Agent():
    ## self, FLAGS, env, agent_params, dims, params, clip_return 원래 input
    ## FLAGS, env,agent_params, policy, dims, logger, monitor=True, **rollout_params
    def __init__(self, dims, FLAGS, env, policy, agent_params, rollout_params, eval_params, monitor=True):

        # print("@ Agent, FLAGS={}, dims={}, env={}, policy={}, agent_params={}, rollout_params={}, eval_params={}".format(FLAGS, dims, env, policy, agent_params, rollout_params, eval_params))

        self.FLAGS = FLAGS
        self.sess = tf.Session()

        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]

        # Create agent with number of levels specified by user       
        self.layers = [Layer(i, dims, FLAGS, env, self.sess, policy, agent_params, rollout_params, eval_params, monitor=True) for i in range(FLAGS.layers)]        

        # Below attributes will be used help save network parameters
        self.saver = None
        self.model_dir = None
        self.model_loc = None

        # Initialize actor/critic networks.  Load saved parameters if not retraining
        # self.initialize_networks()   
        
        # goal_array will store goal for each layer of agent.
        self.goal_array = [None for i in range(FLAGS.layers)]

        self.current_state = None

        # Track number of low-level actions executed
        self.steps_taken = 0

        # Below hyperparameter specifies number of Q-value updates made after each episode
        self.num_updates = 40

        # Below parameters will be used to store performance results
        self.performance_log = []

        self.other_params = agent_params

        ##
        self.goal_space_train = [[-np.pi,np.pi],[-np.pi/4,0],[-np.pi/4,np.pi/4]]
        self.goal_space_test = [[-np.pi,np.pi],[-np.pi/4,0],[-np.pi/4,np.pi/4]]
        self.subgoal_bounds = np.array([[-2*np.pi,2*np.pi],[-2*np.pi,2*np.pi],[-2*np.pi,2*np.pi],[-4,4],[-4,4],[-4,4]])

        self.end_goal_dim = len(self.goal_space_test)
        self.subgoal_dim = len(self.subgoal_bounds)
        # self.subgoal_bounds = subgoal_bounds
        ##

        # return 

        # ## (FLAGS, env, dims, params, clip_return):

        # 원래 있던 코드지만, layer마다 DDPG 에이전트 만들어주기위해 layer.py로 넘겼다.
        # self.policy = config.configure_ddpg(params=params, clip_return=clip_return, FLAGS=FLAGS, agent_params=agent_params, dims=dims)
        
        # # policy = 1
        # return policy
        # ##


    # Determine whether or not each layer's goal was achieved.  Also, if applicable, return the highest level whose goal was achieved.
    def check_goals(self,env):

        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for i in range(self.FLAGS.layers)]

        max_lay_achieved = None

        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = env.project_state_to_subgoal(env.sim, self.current_state)
        proj_end_goal = env.project_state_to_end_goal(env.sim, self.current_state)

        for i in range(self.FLAGS.layers):

            goal_achieved = True
            
            # If at highest layer, compare to end goal thresholds
            if i == self.FLAGS.layers - 1:

                # Check dimensions are appropriate         
                assert len(proj_end_goal) == len(self.goal_array[i]) == len(env.end_goal_thresholds), "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_end_goal)):
                    if np.absolute(self.goal_array[i][j] - proj_end_goal[j]) > env.end_goal_thresholds[j]:
                        goal_achieved = False
                        break

            # If not highest layer, compare to subgoal thresholds
            else:

                # Check that dimensions are appropriate
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(env.subgoal_thresholds), "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"           

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_subgoal)):
                    if np.absolute(self.goal_array[i][j] - proj_subgoal[j]) > env.subgoal_thresholds[j]:
                        goal_achieved = False
                        break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False
            

        return goal_status, max_lay_achieved


    def initialize_networks(self):

        model_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(model_vars)

        # Set up directory for saving models
        self.model_dir = os.getcwd() + '/models'
        self.model_loc = self.model_dir + '/HAC.ckpt'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

         # Initialize actor/critic networks
        self.sess.run(tf.global_variables_initializer())

        # If not retraining, restore weights
        # if we are not retraining from scratch, just restore weights
        if self.FLAGS.retrain == False:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))


    # Save neural network parameters
    def save_model(self, episode):
        self.saver.save(self.sess, self.model_loc, global_step=episode)


    # Update actor and critic networks for each layer
    def learn(self):

        for i in range(len(self.layers)):   
            self.layers[i].learn(self.num_updates)


    # Function returns an end goal
    def get_next_goal(self, test, layer_num):

        end_goal = np.zeros((len(self.goal_space_test)))

        end_goal = np.zeros(shape=(self.end_goal_dim,))
        for i in range(layer_num):
            end_goal[i] = np.random.uniform(self.goal_space_test[i][0],self.goal_space_test[i][1])
            print("at get_next_goal1={}\t 1={}".format(self.goal_space_test[i][0],self.goal_space_test[i][1]))
            
        # end_goal[1] = np.random.uniform(self.goal_space_test[1][0],self.goal_space_test[1][1])
        # end_goal[2] = np.random.uniform(self.goal_space_test[2][0],self.goal_space_test[2][1])


        if not test and self.goal_space_train is not None:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_train[i][0],self.goal_space_train[i][1])
        else:
            assert self.goal_space_test is not None, "Need goal space for testing. Set goal_space_test variable in \"design_env.py\" file"

            for i in range(len(self.goal_space_test)):
                end_goal[i] = np.random.uniform(self.goal_space_test[i][0],self.goal_space_test[i][1])


        # # Visualize End Goal
        # self.display_end_goal(end_goal)

        return end_goal
    # def train_original(self,env, episode_num):

    #     # Select final goal from final goal space, defined in "design_agent_and_env.py" 
    #     self.goal_array[self.FLAGS.layers - 1] = env.get_next_goal(self.FLAGS.test, self.FLAGS.layers)
    #     print("Next End Goal: ", self.goal_array[self.FLAGS.layers - 1])

    #     # Select initial state from in initial state space, defined in environment.py
    #     # self.current_state = env.reset_sim()
    #     self.current_state = env.reset().observation
    #     # print("Initial State: ", self.current_state)

    #     # Reset step counter
    #     self.steps_taken = 0

    #     # Train for an episode
    #     # ?? FLAGS.layers-1 면 최상단 layer를 episode만큼 train한다?
    #     print("@ agent.train, layer-1={}".format(self.FLAGS.layers-1))
    #     goal_status, max_lay_achieved = self.layers[self.FLAGS.layers-1].train(self,env, episode_num = episode_num)

    #     # Update actor/critic networks if not testing
    #     if not self.FLAGS.test:
    #         self.learn()

    #     # Return whether end goal was achieved
    #     return goal_status[self.FLAGS.layers-1]

    # Train agent for an episode
    def train(self, env, episode_num):

        end_goal = np.zeros(shape=(self.end_goal_dim,))
        
        ## from rollout.py
        print("env={}".format(env))
        self.obs_dict = env.reset()
        print("@ agent, obs_dict={}".format(self.obs_dict))
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal']
        ##
        # Select final goal from final goal space, defined in "design_agent_and_env.py" 
        # self.goal_array[self.FLAGS.layers - 1] = self.get_next_goal(self.FLAGS.test, self.FLAGS.layers)
        self.goal_array[self.FLAGS.layers - 1] = self.g[0] ##
        print("Next End Goal: ", self.goal_array[self.FLAGS.layers - 1])

        
        

        # self.goal_array[self.FLAGS.layers - 1] = get_next_goal(self.FLAGS.test, self.FLAGS.layers)
        # self.goal_array[self.FLAGS.layers - 1] = self.g
        # print("Next End Goal: ", self.goal_array[self.FLAGS.layers - 1])

        # Select initial state from in initial state space, defined in environment.py
        # self.current_state = env.reset_sim()
        self.current_state = self.initial_o[0]
        print("Initial State: ", self.current_state)

        # Reset step counter
        self.steps_taken = 0

        # Train for an episode
        goal_status, max_lay_achieved = self.layers[self.FLAGS.layers-1].train(self, env, self.obs_dict, episode_num = episode_num)

        # Update actor/critic networks if not testing
        if not self.FLAGS.test:
            self.learn() # 각 layer learn @layer.py

        # Return whether end goal was achieved
        return goal_status[self.FLAGS.layers-1]

    
    # Save performance evaluations
    def log_performance(self, success_rate):
        
        # Add latest success_rate to list
        self.performance_log.append(success_rate)

        # Save log
        cpickle.dump(self.performance_log,open("performance_log.p","wb"))
        

        

        
        
        



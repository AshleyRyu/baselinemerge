

###DDPG

agent에 들어가는 dictionary 집합은

```dict_items([('obs0', <tf.Tensor 'obs0:0' shape=(?, 16) dtype=float32>), ('obs1', <tf.Tensor 'obs1:0' shape=(?, 16) dtype=float32>), ('terminals1', <tf.Tensor 'terminals1:0' shape=(?, 1) dtype=float32>), ('rewards', <tf.Tensor 'rewards:0' shape=(?, 1) dtype=float32>), ('actions', <tf.Tensor 'actions:0' shape=(?, 4) dtype=float32>), ('critic_target', <tf.Tensor 'critic_target:0' shape=(?, 1) dtype=float32>), ('param_noise_stddev', <tf.Tensor 'param_noise_stddev:0' shape=() dtype=float32>), ('gamma', 0.99), ('tau', 0.01), ('memory', <baselines.ddpg.memory.Memory object at 0x1c2efb82e8>), ('normalize_observations', True), ('normalize_returns', False), ('action_noise', None), ('param_noise', AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2, adoption_coefficient=1.01)), ('action_range', (-1.0, 1.0)), ('return_range', (-inf, inf)), ('observation_range', (-5.0, 5.0)), ('critic', <baselines.ddpg.models.Critic object at 0x1c2f8fcb38>), ('actor', <baselines.ddpg.models.Actor object at 0x1c2f8fcb70>), ('actor_lr', 0.0001), ('critic_lr', 0.001), ('clip_norm', None), ('enable_popart', False), ('reward_scale', 1.0), ('batch_size', 64), ('stats_sample', None), ('critic_l2_reg', 0.01), ('obs_rms', <baselines.common.mpi_running_mean_std.RunningMeanStd object at 0x1c305a7630>), ('ret_rms', None), ('target_actor', <baselines.ddpg.models.Actor object at 0x1c2f8fce80>), ('target_critic', <baselines.ddpg.models.Critic object at 0x1c2f8fcf60>), ('actor_tf', <tf.Tensor 'actor/Tanh_2:0' shape=(?, 4) dtype=float32>), ('normalized_critic_tf', <tf.Tensor 'critic/output/BiasAdd:0' shape=(?, 1) dtype=float32>), ('critic_tf', <tf.Tensor 'clip_by_value_2:0' shape=(?, 1) dtype=float32>), ('normalized_critic_with_actor_tf', <tf.Tensor 'critic_1/output/BiasAdd:0' shape=(?, 1) dtype=float32>), ('critic_with_actor_tf', <tf.Tensor 'clip_by_value_3:0' shape=(?, 1) dtype=float32>), ('target_Q', <tf.Tensor 'add:0' shape=(?, 1) dtype=float32>), ('perturbed_actor_tf', <tf.Tensor 'param_noise_actor/Tanh_2:0' shape=(?, 4) dtype=float32>), ('perturb_policy_ops', <tf.Operation 'group_deps' type=NoOp>), ('perturb_adaptive_policy_ops', <tf.Operation 'group_deps_1' type=NoOp>), ('adaptive_policy_distance', <tf.Tensor 'Sqrt:0' shape=() dtype=float32>), ('actor_loss', <tf.Tensor 'Neg:0' shape=() dtype=float32>), ('actor_grads', <tf.Tensor 'concat:0' shape=(5508,) dtype=float32>), ('actor_optimizer', <baselines.common.mpi_adam.MpiAdam object at 0x1c4bd42358>), ('critic_loss', <tf.Tensor 'add_13:0' shape=() dtype=float32>), ('critic_grads', <tf.Tensor 'concat_2:0' shape=(5569,) dtype=float32>), ('critic_optimizer', <baselines.common.mpi_adam.MpiAdam object at 0x1c4c1a2f60>), ('stats_ops', [<tf.Tensor 'Mean_3:0' shape=() dtype=float32>, <tf.Tensor 'Mean_4:0' shape=() dtype=float32>, <tf.Tensor 'Mean_5:0' shape=() dtype=float32>, <tf.Tensor 'Sqrt_1:0' shape=() dtype=float32>, <tf.Tensor 'Mean_8:0' shape=() dtype=float32>, <tf.Tensor 'Sqrt_2:0' shape=() dtype=float32>, <tf.Tensor 'Mean_11:0' shape=() dtype=float32>, <tf.Tensor 'Sqrt_3:0' shape=() dtype=float32>, <tf.Tensor 'Mean_14:0' shape=() dtype=float32>, <tf.Tensor 'Sqrt_4:0' shape=() dtype=float32>]), ('stats_names', ['obs_rms_mean', 'obs_rms_std', 'reference_Q_mean', 'reference_Q_std', 'reference_actor_Q_mean', 'reference_actor_Q_std', 'reference_action_mean', 'reference_action_std', 'reference_perturbed_action_mean', 'reference_perturbed_action_std']), ('target_init_updates', [<tf.Operation 'group_deps_4' type=NoOp>, <tf.Operation 'group_deps_6' type=NoOp>]), ('target_soft_updates', [<tf.Operation 'group_deps_5' type=NoOp>, <tf.Operation 'group_deps_7' type=NoOp>]), ('initial_state', None)])```

1st trial 경과
```
--------------------------------------
| obs_rms_mean            | 0.474    |
| obs_rms_std             | 0.154    |
| param_noise_stddev      | 0.134    |
| reference_action_mean   | 0.0221   |
| reference_action_std    | 0.684    |
| reference_actor_Q_mean  | -8.53    |
| reference_actor_Q_std   | 3.75     |
| reference_perturbed_... | 0.621    |
| reference_Q_mean        | -8.51    |
| reference_Q_std         | 3.76     |
| rollout/actions_mean    | -0.0503  |
| rollout/actions_std     | 0.652    |
| rollout/episode_steps   | 50       |
| rollout/episodes        | 80       |
| rollout/Q_mean          | -4.7     |
| rollout/return          | -49.9    |
| rollout/return_history  | -49.9    |
| total/duration          | 21.3     |
| total/episodes          | 80       |
| total/epochs            | 2        |
| total/steps             | 4000     |
| total/steps_per_second  | 188      |
| train/loss_actor        | 8.55     |
| train/loss_critic       | 1.45e+06 |
| train/param_noise_di... | 0.556    |
--------------------------------------
```





###TD3

agent에 들어가는 dictionary 집합은

```dict_items([('obs0', <tf.Tensor 'obs0:0' shape=(?, 16) dtype=float32>), ('obs1', <tf.Tensor 'obs1:0' shape=(?, 16) dtype=float32>), ('terminals1', <tf.Tensor 'terminals1:0' shape=(?, 1) dtype=float32>), ('rewards', <tf.Tensor 'rewards:0' shape=(?, 1) dtype=float32>), ('actions', <tf.Tensor 'actions:0' shape=(?, 4) dtype=float32>), ('critic_target', <tf.Tensor 'critic_target:0' shape=(?, 1) dtype=float32>), ('param_noise_stddev', <tf.Tensor 'param_noise_stddev:0' shape=() dtype=float32>), ('gamma', 0.99), ('tau', 0.01), ('memory', <baselines.ddpg.memory.Memory object at 0x1c2dcfc438>), ('normalize_observations', True), ('normalize_returns', False), ('action_noise', None), ('param_noise', AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2, adoption_coefficient=1.01)), ('action_range', (-1.0, 1.0)), ('return_range', (-inf, inf)), ('observation_range', (-5.0, 5.0)), ('critic', <baselines.ddpg.models.Critic object at 0x1c2e699eb8>), ('actor', <baselines.ddpg.models.Actor object at 0x1c2e699ef0>), ('actor_lr', 0.0001), ('critic_lr', 0.001), ('clip_norm', None), ('enable_popart', False), ('reward_scale', 1.0), ('batch_size', 64), ('stats_sample', None), ('critic_l2_reg', 0.01), ('td3_variant', False), ('td3_policy_freq', 1), ('td3_policy_noise', 0.0), ('td3_noise_clip', 0.5), ('obs_rms', <baselines.common.mpi_running_mean_std.RunningMeanStd object at 0x1c2f3779b0>), ('ret_rms', None), ('target_actor', <baselines.ddpg.models.Actor object at 0x1c2e699fd0>), ('target_critic', <baselines.ddpg.models.Critic object at 0x1c2f393cf8>), ('actor_tf', <tf.Tensor 'actor/Tanh_2:0' shape=(?, 4) dtype=float32>), ('normalized_critic_tf', <tf.Tensor 'critic/dense/BiasAdd:0' shape=(?, 1) dtype=float32>), ('critic_tf', <tf.Tensor 'clip_by_value_2:0' shape=(?, 1) dtype=float32>), ('normalized_critic_with_actor_tf', <tf.Tensor 'critic_1/dense/BiasAdd:0' shape=(?, 1) dtype=float32>), ('critic_with_actor_tf', <tf.Tensor 'clip_by_value_3:0' shape=(?, 1) dtype=float32>), ('target_Q', <tf.Tensor 'add:0' shape=(?, 1) dtype=float32>), ('perturbed_actor_tf', <tf.Tensor 'param_noise_actor/Tanh_2:0' shape=(?, 4) dtype=float32>), ('perturb_policy_ops', <tf.Operation 'group_deps' type=NoOp>), ('perturb_adaptive_policy_ops', <tf.Operation 'group_deps_1' type=NoOp>), ('adaptive_policy_distance', <tf.Tensor 'Sqrt:0' shape=() dtype=float32>), ('actor_loss', <tf.Tensor 'Neg:0' shape=() dtype=float32>), ('actor_grads', <tf.Tensor 'concat:0' shape=(5508,) dtype=float32>), ('actor_optimizer', <baselines.common.mpi_adam.MpiAdam object at 0x1c4af73f60>), ('critic_loss', <tf.Tensor 'add_13:0' shape=() dtype=float32>), ('critic_grads', <tf.Tensor 'concat_2:0' shape=(5569,) dtype=float32>), ('critic_optimizer', <baselines.common.mpi_adam.MpiAdam object at 0x1c4af80358>), ('stats_ops', [<tf.Tensor 'Mean_3:0' shape=() dtype=float32>, <tf.Tensor 'Mean_4:0' shape=() dtype=float32>, <tf.Tensor 'Mean_5:0' shape=() dtype=float32>, <tf.Tensor 'Sqrt_1:0' shape=() dtype=float32>, <tf.Tensor 'Mean_8:0' shape=() dtype=float32>, <tf.Tensor 'Sqrt_2:0' shape=() dtype=float32>, <tf.Tensor 'Mean_11:0' shape=() dtype=float32>, <tf.Tensor 'Sqrt_3:0' shape=() dtype=float32>, <tf.Tensor 'Mean_14:0' shape=() dtype=float32>, <tf.Tensor 'Sqrt_4:0' shape=() dtype=float32>]), ('stats_names', ['obs_rms_mean', 'obs_rms_std', 'reference_Q_mean', 'reference_Q_std', 'reference_actor_Q_mean', 'reference_actor_Q_std', 'reference_action_mean', 'reference_action_std', 'reference_perturbed_action_mean', 'reference_perturbed_action_std']), ('target_init_updates', [<tf.Operation 'group_deps_4' type=NoOp>, <tf.Operation 'group_deps_6' type=NoOp>]), ('actor_target_soft_updates', <tf.Operation 'group_deps_5' type=NoOp>), ('critic_target_soft_updates', <tf.Operation 'group_deps_7' type=NoOp>), ('initial_state', None)])```


1st trial 경과

```
--------------------------------------
| obs_rms_mean            | 0.475    |
| obs_rms_std             | 0.172    |
| param_noise_stddev      | 0.134    |
| reference_action_mean   | 0.0291   |
| reference_action_std    | 0.685    |
| reference_actor_Q_mean  | -6.29    |
| reference_actor_Q_std   | 2.14     |
| reference_perturbed_... | 0.719    |
| reference_Q_mean        | -6.68    |
| reference_Q_std         | 1.93     |
| rollout/actions_mean    | -0.0189  |
| rollout/actions_std     | 0.71     |
| rollout/episode_steps   | 50       |
| rollout/episodes        | 80       |
| rollout/Q_mean          | -2.93    |
| rollout/return          | -49.9    |
| rollout/return_history  | -49.9    |
| total/duration          | 22       |
| total/episodes          | 80       |
| total/epochs            | 2        |
| total/steps             | 4000     |
| total/steps_per_second  | 182      |
| train/loss_actor        | 6.22     |
| train/loss_critic       | 1.23     |
| train/param_noise_di... | 0.633    |
--------------------------------------
```

로그를 예쁘게 찍어보겠습니다..

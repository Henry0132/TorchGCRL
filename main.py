import argparse
import torch
import os
import pickle
import numpy as np
import time
import gym
import random
import ogbench
import tqdm
from collections import defaultdict
from utils.log_utils import CsvLogger

# from env_utils import make_env_and_datasets

from utils.utils import Dataset, ReplayBuffer, GCDataset, HGCDataset

def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def eval_policy(agent, env, num_eval_episodes=10, num_video_episodes=0, video_frame_skip=3, device=None):
    trajs = []
    stats = defaultdict(list)
    renders = []
    for i in tqdm.trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []

        while not done:
            action = agent.sample_action(state=observation, goal=goal, device=device)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)
                
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
                )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)
        
    return stats, trajs, renders


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="humanoidmaze-large-navigate-v0", help="Environment (dataset) name.")
    parser.add_argument("--seed", type=int, default=0, help="The seeds were selected to ensure the reproducibility of the experiment")
    parser.add_argument("--device", type=str, default="cuda:0", help="Which device you want")
    parser.add_argument("--algo", type=str, default='GCDiffusionQL', help="Which algo you want to select")
    parser.add_argument("--dataset_class", type=str, default='GCDataset', help="Dataset class include 'ReplayBuffer', 'GCDataset' and 'HGCDataset'.")
    
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--tau", type=float, default=0.005, help="The rate for update target networks")

    # Diffusion
    parser.add_argument("--beta_schedule", type=str, default='vp', help="linear, cosine, vp")  
    parser.add_argument("--T", type=int, default=10, help="Number of diffusion timesteps")

    parser.add_argument("--step_start_ema",type=int, default=1000, help="EMA model start step")
    parser.add_argument("--ema_decay", type=float, default=0.995, help="The rate of EMA model decay")
    parser.add_argument("--update_ema_every", type=int, default=5, help="The frequency of EMA model updating")
    parser.add_argument("--lr_decay", type=bool, default=False, help="choose the learning rate decay")
    parser.add_argument("--gn", type=float, default=1.0, help="grad norm")

    parser.add_argument("--alpha", type=float, default=2.5, help="q_loss weight")

    parser.add_argument("--offline_steps", type=int, default=1000000, help="Number of offline steps.")
    parser.add_argument("--online_steps", type=int, default=0, help="Number of online steps.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--buffer_size", type=int, default=2000000, help="Replay buffer size.")
    parser.add_argument("--log_interval", type=int, default=5000, help="Logging interval.")
    parser.add_argument("--eval_interval", type=int, default=10000, help="Evaluation interval.")
    parser.add_argument("--save_interval", type=int, default=1000000, help="Saving interval.")
    parser.add_argument("--save_dir", type=str, default="./result", help="Saving path.")

    parser.add_argument("--eval_tasks", type=int, default=None, help="Number of tasks to evaluate (None for all).")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes to evaluate the agent.")
    parser.add_argument("--video_episodes", type=int, default=0, help="Number of video episodes for each task.")
    parser.add_argument("--video_frame_skip", type=int, default=3, help="Frame skip for videos.")

    parser.add_argument("--p_aug", type=str, default=None, help="Probability of applying image augmentation.")
    parser.add_argument("--frame_stack", type=int, default=None, help="Number of frames to stack.")
    parser.add_argument("--balanced_sampling", type=bool, default=False, help="Whether to use balanced sampling for online fine-tuning.")
    parser.add_argument("--value_p_curgoal", type=float, default=0.0, help="Unused (defined for compatibility with GCDataset).")
    parser.add_argument("--value_p_trajgoal", type=float, default=1.0, help="Unused (defined for compatibility with GCDataset).")
    parser.add_argument("--value_p_randomgoal", type=float, default=0.0, help="Unused (defined for compatibility with GCDataset).")
    parser.add_argument("--value_geom_sample", type=bool, default=False, help="Unused (defined for compatibility with GCDataset).")
    parser.add_argument("--actor_p_curgoal", type=float, default=0.0, help="Probability of using the current state as the actor goal.")
    parser.add_argument("--actor_p_trajgoal", type=float, default=1.0, help="Probability of using a future state in the same trajectory as the actor goal.")
    parser.add_argument("--actor_p_randomgoal", type=float, default=0.0, help="Probability of using a random state as the actor goal.")
    parser.add_argument("--actor_geom_sample", type=bool, default=False, help="Whether to use geometric sampling for future actor goals.")
    parser.add_argument("--gc_negative", type=bool, default=True, help="Unused (defined for compatibility with GCDataset).")
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.algo}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    # Make environment and datasets.
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(args.env)
    if args.video_episodes > 0:
        assert 'singletask' in args.env, 'Rendering is currently only supported for OGBench environments.'
    if args.online_steps > 0:
        assert 'visual' not in args.env, 'Online fine-tuning is currently not supported for visual environments.'

    # Initialize agent.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set up datasets
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    dataset_kwargs = dict(
        gc_negative=args.gc_negative,
        discount=args.discount,
        value_p_curgoal=args.value_p_curgoal,
        value_p_trajgoal=args.value_p_trajgoal,
        value_p_randomgoal=args.value_p_randomgoal,
        value_geom_sample=args.value_geom_sample,
        actor_p_curgoal=args.actor_p_curgoal,
        actor_p_trajgoal=args.actor_p_trajgoal,
        actor_p_randomgoal=args.actor_p_randomgoal,
        actor_geom_sample=args.actor_geom_sample,
        p_aug=args.p_aug,
        frame_stack=args.frame_stack,
    )

    # for no goal-conditional algo:
    if args.dataset_class == 'ReplayBuffer':
        if args.balanced_sampling:
            # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
            example_transition = {k: v[0] for k, v in train_dataset.items()}
            replay_buffer = ReplayBuffer.create(example_transition, size=args.buffer_size)
        else:
            # Use the training dataset as the replay buffer.
            train_dataset = ReplayBuffer.create_from_initial_dataset(
                dict(train_dataset), size=max(args.buffer_size, train_dataset.size + 1)
            )
            replay_buffer = train_dataset
        # Set p_aug and frame_stack.
        for dataset in [train_dataset, val_dataset, replay_buffer]:
            if dataset is not None:
                dataset.p_aug = args.p_aug
                dataset.frame_stack = args.frame_stack
                if args.algo == 'rebrac':
                    dataset.return_next_actions = True

    # for goal-conditional algo:
    elif args.dataset_class == 'GCDataset':
        train_dataset = GCDataset(train_dataset, **dataset_kwargs)
        val_dataset = GCDataset(val_dataset, **dataset_kwargs)
    elif args.dataset_class == 'HGCDataset':
        train_dataset = HGCDataset(train_dataset, **dataset_kwargs)
        val_dataset = HGCDataset(val_dataset, **dataset_kwargs)

    example_batch = train_dataset.sample(1)

    state_dim = example_batch['observations'].shape[1]
    goal_dim = example_batch['observations'].shape[1]
    action_dim = example_batch['actions'].shape[1]
    max_action = env.action_space.high[0]

    # Create agent.
    if args.algo == 'GCBC':
        from algos.GCBC import GCBC as Agent
        agent = Agent(
            state_dim=state_dim, 
            goal_dim=goal_dim, 
            action_dim=action_dim, 
            max_action=max_action, 
            device=args.device, 
            lr=3e-4, 
            hidden_dim=256
        )
    elif args.algo == 'GCTD3BC':
        from algos.GCTD3BC import GCTD3BC as Agent
        agent = Agent(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=args.device,
            lr=3e-4,
            discount=args.discount,
            tau=args.tau,
            hidden_dim=256,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            alpha=args.alpha
        )
    elif args.algo == "GCDiffusionQL":
        from algos.GCDiffusionQL import GCDiffusionQL as Agent
        agent = Agent(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=args.device,
            lr=3e-4,
            discount=args.discount,
            tau=args.tau,
            hidden_dim=256,
            alpha=args.alpha,
            T=args.T,
            beta_schedule=args.beta_schedule,
            step_start_ema=args.step_start_ema,
            ema_decay=args.ema_decay,
            update_ema_every=args.update_ema_every,
            lr_decay=args.lr_decay,
            grad_norm=args.gn
        )
    else:
        pass
    
    save_path = f'{args.save_dir}/{args.algo}/{args.env}'
    os.makedirs(save_path, exist_ok=True)
    train_logger = CsvLogger(os.path.join(save_path, f'train_{args.alpha}.csv'))
    eval_logger = CsvLogger(os.path.join(save_path, f'eval_{args.alpha}.csv'))
    first_time = time.time()
    last_time = time.time()

    for t in tqdm.trange(int(args.offline_steps)):
        losses = agent.train(train_dataset, args.batch_size)
        if (t+1) % args.log_interval == 0:
            loss_str = ' | '.join(f"{k}: {v:.4f}" for k, v in losses.items())
            print(f"Step {t+1}: {loss_str}")
        if (t+1) % args.eval_interval == 0:
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = args.eval_tasks if args.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = eval_policy(
                    agent=agent, 
                    env=env, 
                    num_eval_episodes=args.eval_episodes, 
                    num_video_episodes=args.video_episodes, 
                    video_frame_skip=args.video_frame_skip,
                    device=args.device
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            eval_logger.log(eval_metrics, step=t+1)

        env.close()

            # if args.video_episodes > 0:
            #     video = get_wandb_video(renders=renders, n_cols=num_tasks)
            #     eval_metrics['video'] = video


            

    

    





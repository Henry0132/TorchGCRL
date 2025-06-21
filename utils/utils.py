import dataclasses
import numpy as np
import torch
import copy
import time
import math


def print_banner(s, separator="-", num_star=60):
	print(separator * num_star, flush=True)
	print(s, flush=True)
	print(separator * num_star, flush=True)


class Progress:

	def __init__(self, total, name='Progress', ncol=3, max_length=20, indent=0, line_width=100, speed_update_freq=100):
		self.total = total
		self.name = name
		self.ncol = ncol
		self.max_length = max_length
		self.indent = indent
		self.line_width = line_width
		self._speed_update_freq = speed_update_freq

		self._step = 0
		self._prev_line = '\033[F'
		self._clear_line = ' ' * self.line_width

		self._pbar_size = self.ncol * self.max_length
		self._complete_pbar = '#' * self._pbar_size
		self._incomplete_pbar = ' ' * self._pbar_size

		self.lines = ['']
		self.fraction = '{} / {}'.format(0, self.total)

		self.resume()

	def update(self, description, n=1):
		self._step += n
		if self._step % self._speed_update_freq == 0:
			self._time0 = time.time()
			self._step0 = self._step
		self.set_description(description)

	def resume(self):
		self._skip_lines = 1
		print('\n', end='')
		self._time0 = time.time()
		self._step0 = self._step

	def pause(self):
		self._clear()
		self._skip_lines = 1

	def set_description(self, params=[]):

		if type(params) == dict:
			params = sorted([
				(key, val)
				for key, val in params.items()
			])

		############
		# Position #
		############
		self._clear()

		###########
		# Percent #
		###########
		percent, fraction = self._format_percent(self._step, self.total)
		self.fraction = fraction

		#########
		# Speed #
		#########
		speed = self._format_speed(self._step)

		##########
		# Params #
		##########
		num_params = len(params)
		nrow = math.ceil(num_params / self.ncol)
		params_split = self._chunk(params, self.ncol)
		params_string, lines = self._format(params_split)
		self.lines = lines

		description = '{} | {}{}'.format(percent, speed, params_string)
		print(description)
		self._skip_lines = nrow + 1

	def append_description(self, descr):
		self.lines.append(descr)

	def _clear(self):
		position = self._prev_line * self._skip_lines
		empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])
		print(position, end='')
		print(empty)
		print(position, end='')

	def _format_percent(self, n, total):
		if total:
			percent = n / float(total)

			complete_entries = int(percent * self._pbar_size)
			incomplete_entries = self._pbar_size - complete_entries

			pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]
			fraction = '{} / {}'.format(n, total)
			string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent * 100))
		else:
			fraction = '{}'.format(n)
			string = '{} iterations'.format(n)
		return string, fraction

	def _format_speed(self, n):
		num_steps = n - self._step0
		t = time.time() - self._time0
		speed = num_steps / t
		string = '{:.1f} Hz'.format(speed)
		if num_steps > 0:
			self._speed = string
		return string

	def _chunk(self, l, n):
		return [l[i:i + n] for i in range(0, len(l), n)]

	def _format(self, chunks):
		lines = [self._format_chunk(chunk) for chunk in chunks]
		lines.insert(0, '')
		padding = '\n' + ' ' * self.indent
		string = padding.join(lines)
		return string, lines

	def _format_chunk(self, chunk):
		line = ' | '.join([self._format_param(param) for param in chunk])
		return line

	def _format_param(self, param):
		k, v = param
		return '{} : {}'.format(k, v)[:self.max_length]

	def stamp(self):
		if self.lines != ['']:
			params = ' | '.join(self.lines)
			string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)
			self._clear()
			print(string, end='\n')
			self._skip_lines = 1
		else:
			self._clear()
			self._skip_lines = 0

	def close(self):
		self.pause()


class Silent:

	def __init__(self, *args, **kwargs):
		pass

	def __getattr__(self, attr):
		return lambda *args: None


class EarlyStopping(object):
	def __init__(self, tolerance=5, min_delta=0):
		self.tolerance = tolerance
		self.min_delta = min_delta
		self.counter = 0
		self.early_stop = False

	def __call__(self, train_loss, validation_loss):
		if (validation_loss - train_loss) > self.min_delta:
			self.counter += 1
			if self.counter >= self.tolerance:
				return True
		else:
			self.counter = 0
		return False


def to_torch(batch, device):
    states = torch.from_numpy(batch['observations']).float().to(device)
    actions = torch.from_numpy(batch['actions']).float().to(device)
    terminals = torch.from_numpy(batch['terminals']).float().to(device)
    next_states = torch.from_numpy(batch['next_observations']).float().to(device)
    rewards = torch.from_numpy(batch['rewards']).float().to(device)
    masks = torch.from_numpy(batch['masks']).float().to(device)
    value_goals = torch.from_numpy(batch['value_goals']).float().to(device)
    actor_goals = torch.from_numpy(batch['actor_goals']).float().to(device)

    return states, actions, next_states, rewards, value_goals, actor_goals, terminals, masks


def get_size(data):
    '''Return the size of the dataset'''
    return max(len(arr) for arr in tree_leaves(data))

def tree_map(fn, *trees):
    '''Recursively apply a function to a dict'''
    if isinstance(trees[0], dict):
        return {k: tree_map(fn, *(t[k] for t in trees)) for k in trees[0]}
    else:
        return fn(*trees)
    
def tree_leaves(tree):
    '''Get all leaves (arrays) from a nested dict'''
    if isinstance(tree, dict):
        return sum([tree_leaves(v) for v in tree.values()], [])
    else:
        return [tree]
    
def random_crop(img, crop_from, padding):
    '''Random crop of a single image using torch'''
    img = torch.from_numpy(img)
    padded = torch.nn.functional.pad(img.permute(2, 0, 1), (padding, padding, padding, padding), mode='replicate')
    x, y = crop_from
    cropped = padded[:, y:y + img.shape[0], x:x + img.shape[1]]
    return cropped.permute(1, 2, 0).numpy()

def batched_random_crop(imgs, crop_froms, padding):
    '''Batch version of random_crop'''
    return np.stack([
        random_crop(img, crop_from, padding)
        for img, crop_from in zip(imgs, crop_froms)
    ])

class Dataset:
    def __init__(self, data):
        self._dict = data
        self.size = get_size(data)
        self.frame_stack = None     # Number of frames to stack; set outside the class.
        self.p_aug = None           # Image augmentation probability; set outside the class.
        self.return_next_actions = False    # Whether to additionally return next actions; set outside the class.

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(data['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def copy(self):
        return Dataset(copy.deepcopy(self._dict))
    
    def __getitem__(self, key):
        return self._dict[key]

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    @classmethod
    def create(cls, freeze=True, **fields):
        if freeze:
            tree_map(lambda arr: arr.setflags(write=False), fields)
        return cls(fields)
    
    def get_random_idxs(self, num_idxs):
        return np.random.randint(0, self.size, size=num_idxs)
    
    def get_subset(self, idxs):
        result = tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result
    
    def augment(self, batch, keys):
        padding = 3
        sample_arr = next(iter(batch[keys[0]].values())) if isinstance(batch[keys[0]], dict) else batch[keys[0]]
        batch_size = sample_arr.shape[0]
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        for key in keys:
            batch[key] = tree_map(
                lambda arr: batched_random_crop(arr, crop_froms, padding) if arr.ndim == 4 else arr,
                batch[key]
            )

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)

        if self.frame_stack is not None:
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs, next_obs = [], []
            for i in reversed(range(self.frame_stack)):
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(tree_map(lambda arr: arr[cur_idxs], self._dict['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(tree_map(lambda arr: arr[cur_idxs], self._dict['observations']))
            next_obs.append(tree_map(lambda arr: arr[idxs], self._dict['next_observations']))

            batch['observations'] = tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        
        if self.p_aug is not None and np.random.rand() < self.p_aug:
            self.augment(batch, ['observations', 'next_observations'])

        return batch


class ReplayBuffer(Dataset):
    @classmethod
    def create(cls, transition, size):
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)
        
        buffer_dict = tree_map(create_buffer, transition)
        return cls(buffer_dict)
    
    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[:len(init_buffer)] = init_buffer
            return buffer
        
        buffer_dict = tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        assert set(transition.keys()) == set(self._dict.keys()), "Mismatch in transition keys"
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element
        tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        self.size = self.pointer = 0

@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """
    def __init__(
        self,
        dataset,  # Dataset include ['observations', 'terminals', 'sample', 'size', 'get_random_idxs']
        gc_negative=False,
        discount=0.99,
        value_p_curgoal=0.3,
        value_p_trajgoal=0.3,
        value_p_randomgoal=0.4,
        value_geom_sample=True,
        actor_p_curgoal=0.3,
        actor_p_trajgoal=0.3,
        actor_p_randomgoal=0.4,
        actor_geom_sample=True,
        p_aug=None,
        frame_stack=None,
        preprocess_frame_stack=False,
    ):
        self.dataset = dataset
        self.size = self.dataset.size
        self.gc_negative = gc_negative
        self.discount = discount
        self.value_p_curgoal = value_p_curgoal
        self.value_p_trajgoal = value_p_trajgoal
        self.value_p_randomgoal = value_p_randomgoal
        self.value_geom_sample = value_geom_sample
        self.actor_p_curgoal = actor_p_curgoal
        self.actor_p_trajgoal = actor_p_trajgoal
        self.actor_p_randomgoal = actor_p_randomgoal
        self.actor_geom_sample = actor_geom_sample
        self.p_aug = float(p_aug) if p_aug is not None else None
        self.frame_stack = int(frame_stack) if frame_stack is not None else None
        self.preprocess_frame_stack = preprocess_frame_stack

        # Pre-compute trajectory boundaries.
        self.terminal_locs = np.nonzero(self.dataset['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Pre-compute frame stack
        if self.frame_stack is not None and self.preprocess_frame_stack:
            stacked_obs = self.get_stacked_observations(np.arange(self.size))
            self.dataset['observations'] = stacked_obs

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        if self.frame_stack is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        # Sample goals
        value_goal_idxs = self.sample_goals(idxs, self.value_p_curgoal, self.value_p_trajgoal,
                                            self.value_p_randomgoal, self.value_geom_sample)
        actor_goal_idxs = self.sample_goals(idxs, self.actor_p_curgoal, self.actor_p_trajgoal,
                                            self.actor_p_randomgoal, self.actor_geom_sample)
        
        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(np.float32)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.gc_negative else 0.0)

        # Augmentation
        if self.p_aug is not None and not evaluation:
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch
    
    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        batch_size = len(idxs)

        idxs = np.clip(idxs, 0, self.size - 1)
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Compute final state per sample
        indices = np.searchsorted(self.terminal_locs, idxs)
        indices = np.clip(indices, 0, len(self.terminal_locs) - 1)
        final_state_idxs = self.terminal_locs[indices]

        if geom_sample:
            offsets = np.random.geometric(p=1 - self.discount, size=batch_size)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)
            traj_goal_idxs = np.round((idxs + 1) * distances + final_state_idxs * (1 - distances)).astype(int)

        goal_idxs = np.where(
            np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-6),
            traj_goal_idxs,
            random_goal_idxs
        )

        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)
        goal_idxs = np.clip(goal_idxs, 0, self.size - 1)

        return goal_idxs
    
    def augment(self, batch, keys, padding=4):
        # Basic random crop (PyTorch-like)
        for key in keys:
            imgs = batch[key]
            bs, c, h, w = imgs.shape
            pad_imgs = torch.nn.functional.pad(torch.tensor(imgs), (padding,) * 4, mode='replicate')
            crop_x = torch.randint(0, 2 * padding + 1, (bs,))
            crop_y = torch.randint(0, 2 * padding + 1, (bs,))
            cropped = torch.stack([
                pad_imgs[i, :, y:y + h, x:x + w]
                for i, (x, y) in enumerate(zip(crop_x, crop_y))
            ])
            batch[key] = cropped.numpy()

    def get_observations(self, idxs):
        if self.frame_stack is None or self.preprocess_frame_stack:
            return self.dataset['observations'][idxs]
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Stack past `frame_stack` frames"""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        obs = self.dataset['observations']
        stacked = []
        for i in reversed(range(self.frame_stack)):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            stacked.append(obs[cur_idxs])
        return np.concatenate(stacked, axis=1 if obs.ndim == 4 else -1)


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """
    def __init__(self, *args, subgoal_steps=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.subgoal_steps = subgoal_steps

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        if self.frame_stack is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        # 1. Value goals
        value_goal_idxs = self.sample_goals(
            idxs,
            self.value_p_curgoal,
            self.value_p_trajgoal,
            self.value_p_randomgoal,
            self.value_geom_sample
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(np.float32)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.gc_negative else 0.0)

        # 2. Low-level goals: fixed offset forward within episode
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(idxs + self.subgoal_steps, final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)

        # 3. High-level goals and prediction targets
        if self.actor_geom_sample:
            offsets = np.random.geometric(p=1 - self.discount, size=batch_size)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)

        high_traj_target_idxs = np.minimum(idxs + self.subgoal_steps, high_traj_goal_idxs)

        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(idxs + self.subgoal_steps, final_state_idxs)

        pick_random = np.random.rand(batch_size) < self.actor_p_randomgoal
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)

        # 4. Augmentation
        if self.p_aug is not None and not evaluation:
            if np.random.rand() < self.p_aug:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'value_goals',
                        'low_actor_goals',
                        'high_actor_goals',
                        'high_actor_targets'
                    ]
                )

        return batch
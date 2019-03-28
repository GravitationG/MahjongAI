from __future__ import print_function
import os, errno
import numpy as np
import random
import collections
import Tile
from sklearn.preprocessing import normalize

parallel_parameters = {
    "intra_op_parallelism_threads": 8,
    "inter_op_parallelism_threads": 8,
}

scoring_scheme = [
	[0, 0],
	[40, 60],
	[80, 120],
	[160, 240],
	[320, 480],
	[480, 720],
	[640, 960],
	[960, 1440],
	[1280, 1920],
	[1920, 2880],
	[2560, 3840]
]

predictor_hand_format_to_loss = {
	"distrib": "softmax",
	"exist": "sigmoid",
	"raw_count": "squared"
}




def softmax(y):
	max_vals = np.amax(y, axis = 1, keepdims = True)
	y_exp = np.exp(y - max_vals)
	y_sum = np.sum(y_exp, axis = 1, keepdims = True)
	result = y_exp / y_sum
	return result

def split_data(X, y, train_portion, max_valid_cases = 30000):
	n_samples = y.shape[0]
	valid_count = min(int(n_samples*(1 - train_portion)), max_valid_cases)
	if valid_count <= 0:
		raise Exception("Too few samples to split")

	indices = random.sample(range(n_samples), valid_count)
	valid_X = X[indices, :]
	valid_y = y[indices]
	train_X = np.delete(X, indices, axis = 0)
	train_y = np.delete(y, indices, axis = 0)
	return train_X, train_y, valid_X, valid_y

class Dataset:
	def __init__(self, X, y = None, batch_size = 100, repeat = float("inf"), is_shuffle = True):
		if y is not None and X.shape[0] != y.shape[0]:
			raise Exception("The first dimension of X and y must be the same")
			
		self.__batch_size = batch_size
		self.__repeat = repeat
		self.__is_shuffle = is_shuffle
		self.__cur_repeat_count = 0
		self.__X = X
		self.__y = y
		self.__sample_indices = np.asarray([], dtype = np.int)

	def __new_batch(self):
		if self.__cur_repeat_count >= self.__repeat:
			return
		
		self.__cur_repeat_count += 1
		indices = np.arange(self.__X.shape[0], dtype = np.int)

		if self.__is_shuffle:
			np.random.shuffle(indices)
		self.__sample_indices = np.append(self.__sample_indices, indices)

	def next_element(self):
		if self.__sample_indices.shape[0] < self.__batch_size:
			self.__new_batch()

		if self.__sample_indices.shape[0] == 0:
			raise Exception("Data exhausted")

		indices = self.__sample_indices[0:self.__batch_size]
		self.__sample_indices = self.__sample_indices[self.__batch_size:]
		batch_X = self.__X[indices]

		if self.__y is None:
			return batch_X

		batch_y = self.__y[indices]

		return batch_X, batch_y

'''
	Adapted from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
'''

class SumTree(object):
	"""
	This SumTree code is modified version and the original code is from: 
	https://github.com/jaara/AI-blog/blob/master/SumTree.py
	Story the data with it priority in tree and data frameworks.
	"""
	data_pointer = 0

	def __init__(self, capacity):
		self.capacity = capacity  # for all priority values
		self.tree = np.zeros(2 * capacity - 1)
		# [--------------Parent nodes-------------][-------leaves to recode priority-------]
		#			 size: capacity - 1					   size: capacity
		self.data = np.zeros(capacity, dtype=object)  # for all transitions
		# [--------------data frame-------------]
		#			 size: capacity

	def add(self, p, data):
		tree_idx = self.data_pointer + self.capacity - 1
		self.data[self.data_pointer] = data  # update data_frame
		self.update(tree_idx, p)  # update tree_frame

		self.data_pointer += 1
		if self.data_pointer >= self.capacity:  # replace when exceed the capacity
			self.data_pointer = 0

	def update(self, tree_idx, p):
		change = p - self.tree[tree_idx]
		self.tree[tree_idx] = p
		# then propagate the change through tree
		while tree_idx != 0:	# this method is faster than the recursive loop in the reference code
			tree_idx = (tree_idx - 1) // 2
			self.tree[tree_idx] += change

	def get_leaf(self, v):
		"""
		Tree structure and array storage:
		Tree index:
			 0		 -> storing priority sum
			/ \
		  1	 2
		 / \   / \
		3   4 5   6	-> storing priority for transitions
		Array type for storing:
		[0,1,2,3,4,5,6]
		"""
		parent_idx = 0
		while True:	 # the while loop is faster than the method in the reference code
			cl_idx = 2 * parent_idx + 1		 # this leaf's left and right kids
			cr_idx = cl_idx + 1
			if cl_idx >= len(self.tree):		# reach bottom, end search
				leaf_idx = parent_idx
				break
			else:	   # downward search, always search for a higher priority node
				if v <= self.tree[cl_idx]:
					parent_idx = cl_idx
				else:
					v -= self.tree[cl_idx]
					parent_idx = cr_idx

		data_idx = leaf_idx - self.capacity + 1
		return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

	@property
	def total_p(self):
		return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
	"""
	This SumTree code is modified version and the original code is from:
	https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
	"""
	epsilon = 0.01  # small amount to avoid zero priority
	alpha = 0.6  # [0~1] convert the importance of TD error to priority
	beta = 0.4  # importance-sampling, from initial value increasing to 1
	beta_increment_per_sampling = 0.001
	abs_err_upper = 1.  # clipped abs error

	def __init__(self, capacity):
		self.tree = SumTree(capacity)

	def store(self, transition):
		max_p = np.max(self.tree.tree[-self.tree.capacity:])
		if max_p == 0:
			max_p = self.abs_err_upper
		self.tree.add(max_p, transition)   # set the max p for new p

	def sample(self, n):
		b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
		pri_seg = self.tree.total_p / n	   # priority segment
		self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

		min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p	 # for later calculate ISweight
		for i in range(n):
			a, b = pri_seg * i, pri_seg * (i + 1)
			v = np.random.uniform(a, b)
			idx, p, data = self.tree.get_leaf(v)
			prob = p / self.tree.total_p
			ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
			b_idx[i], b_memory[i, :] = idx, data
		return b_idx, b_memory, ISWeights

	def batch_update(self, tree_idx, abs_errors):
		abs_errors += self.epsilon  # convert to abs and avoid 0
		clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
		ps = np.power(clipped_errors, self.alpha)
		for ti, p in zip(tree_idx, ps):
			self.tree.update(ti, p)


def dnn_encode_state(player, neighbors):
	state = np.zeros((9, 34, 1))
	for tile in player.hand:
		state[0, Tile.convert_tile_index(tile), :] += 1

	players = [player] + list(neighbors)
	for i in range(len(players)):
		p = players[i]
		for _, _, tiles in p.fixed_hand:
			for tile in tiles:
				state[1 + 2 * i, Tile.convert_tile_index(tile), :] += 1

		for tile in p.get_discarded_tiles():
			state[2 + 2 * i, Tile.convert_tile_index(tile), :] += 1
	return state


def extended_dnn_encode_state(player, neighbors, new_tile=None, cpk_tile=None):
	state = np.zeros((10, 34, 1))
	for tile in player.hand:
		state[0, Tile.convert_tile_index(tile), :] += 1

	if new_tile is not None:
		state[0, Tile.convert_tile_index(new_tile), :] += 1

	if cpk_tile is not None:
		state[9, Tile.convert_tile_index(cpk_tile), :] += 1

	players = [player] + list(neighbors)
	for i in range(len(players)):
		p = players[i]
		for _, _, tiles in p.fixed_hand:
			for tile in tiles:
				state[1 + 2 * i, Tile.convert_tile_index(tile), :] += 1

		for tile in p.get_discarded_tiles():
			state[2 + 2 * i, Tile.convert_tile_index(tile), :] += 1

	return state


def get_input_list(title, options):
	i = 0
	options_str = ""
	for option in options:
		options_str += "%d. %s\n" % (i, option)
		i += 1
	print("%s\n%s" % (title, options_str), end="")
	while True:
		result = input("Enter your choice [%d - %d]: " % (0, len(options) - 1))
		try:
			result = int(result)
			if result < 0 or result >= len(options):
				raise ValueError
			return result
		except ValueError:
			print("Input must be an integer within the range, try again")


def get_input_range(title, lower_bound, upper_bound, lb_inclusive=True, ub_inclusive=True):
	range_str, lb_sign, ub_sign = "", "", ""
	if lb_inclusive:
		lb_sign = "["
	else:
		lb_sign = "("

	if ub_inclusive:
		ub_sign = "]"
	else:
		ub_sign = ")"

	range_str = "%s%d,%d%s" % (lb_sign, lower_bound, upper_bound, ub_sign)

	while True:
		result = input("%s %s: " % (title, range_str))
		try:
			result = int(result)
			if result < lower_bound or result > upper_bound:
				raise ValueError
			if not lb_inclusive and result == lower_bound:
				raise ValueError
			if not ub_inclusive and result == upper_bound:
				raise ValueError
			return result
		except ValueError:
			print("Input must be an integer within the range, try again")


def map_increment(map, index, increment=1, remove_zero=False):
	if index is None:
		raise Exception("Index cannot be None")

	result = map.get(index, 0) + increment
	map[index] = result

	if remove_zero and result == 0:
		del map[index]

	return map


def map_retrieve(map, index, default_val=0):
	if index is None:
		return default_val

	if not isinstance(index, collections.Hashable):
		index = str(index)

	return map.get(index, default_val)


def print_hand(hand, end="\n"):
	meld_type, is_secret, tiles = None, None, None
	for meld in hand:
		if type(meld) == Tile.Tile:
			print(meld.symbol, end="")
		else:
			if len(meld) == 3:
				meld_type, is_secret, tiles = meld
			elif len(meld) == 2:
				meld_type, tiles = meld
				is_secret = False
			else:
				raise Exception("unexpected structure of hand")
			for tile in tiles:
				print(tile.symbol, end="")
			print(" ", end="")
	print("", end=end)


def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)


def random_choice(objs, p):
	s = 0
	target = np.random.uniform()
	n_item = len(p) if type(p) is list else p.shape[0]
	for i in range(n_item):
		s += p[i]
		if s >= target:
			return objs[i]
	return objs[n_item - 1]


def print_game_board(player_name, fixed_hand, hand, neighbors, game, new_tile=None, print_stolen_tiles=False):
	line_format_left = u"|{next:<20s}|{opposite:<20s}|{prev:<20s}|"
	line_format_right = u"|{next:>20s}|{opposite:>20s}|{prev:>20s}|"
	line_merged_format_left = u"|{msg:<62s}|"
	line_merged_format_right = u"|{msg:>62s}|"

	horizontal_line = line_merged_format_left.format(msg='-' * 62)

	print("Wake up %s!" % player_name)

	print(horizontal_line)
	print(line_merged_format_right.format(msg="Game of %s wind [%d]" % (game.game_wind, game.deck_size)))
	print(horizontal_line)
	print(line_format_left.format(next="Next Player", opposite="Opposite Player", prev="Previous Player"))
	print(line_format_left.format(next="(%s)" % neighbors[0].name, opposite="(%s)" % neighbors[1].name,
								  prev="(%s)" % neighbors[2].name))
	print(horizontal_line)

	fixed_hands_strs = []
	hand_sizes = []
	disposed_tiles_symbols = []
	filter_state = None if print_stolen_tiles else "unstolen"

	for neighbor in neighbors:
		fixed_hand_str = ""
		for meld_type, is_secret, tiles in neighbor.fixed_hand:
			if is_secret:
				fixed_hand_str += Tile.tile_back_symbol + tiles[0].symbol + tiles[0].symbol + Tile.tile_back_symbol
			else:
				fixed_hand_str += "".join([tile.symbol for tile in tiles])
		fixed_hands_strs.append(fixed_hand_str)
		hand_sizes.append(neighbor.hand_size)

		disposed_tiles = neighbor.get_discarded_tiles(filter_state)
		disposed_tiles_symbols.append(''.join([tile.symbol for tile in disposed_tiles]))

	print(line_format_left.format(next=fixed_hands_strs[0], opposite=fixed_hands_strs[1], prev=fixed_hands_strs[2]))
	print(line_format_right.format(next="%s -%d" % (Tile.tile_back_symbol * hand_sizes[0], hand_sizes[0]),
								   opposite="%s -%d" % (Tile.tile_back_symbol * hand_sizes[1], hand_sizes[1]),
								   prev="%s -%d" % (Tile.tile_back_symbol * hand_sizes[2], hand_sizes[2])))

	print(horizontal_line)
	is_continue_print = True

	while is_continue_print:
		print(line_format_left.format(next=disposed_tiles_symbols[0][0:20], opposite=disposed_tiles_symbols[1][0:20],
									  prev=disposed_tiles_symbols[2][0:20]))
		is_continue_print = False
		for i in range(3):
			disposed_tiles_symbols[i] = disposed_tiles_symbols[i][20:]
			if len(disposed_tiles_symbols[i]) > 0:
				is_continue_print = True

	print(horizontal_line)
	print(line_merged_format_left.format(msg="%s's tiles:" % (player_name)))
	fixed_hand_str = ""
	for meld_type, is_secret, tiles in fixed_hand:
		if is_secret:
			fixed_hand_str += "".join([Tile.tile_back_symbol, tiles[0].symbol, tiles[0].symbol, Tile.tile_back_symbol])
		else:
			fixed_hand_str += "".join([tile.symbol for tile in tiles])
	print(line_merged_format_left.format(msg=fixed_hand_str))

	line_1, line_2 = "", ""
	i = 0
	for tile in hand:
		line_1 += "%s  " % (tile.symbol)
		line_2 += "{digit:<3s}".format(digit=str(i))
		i += 1
	print(line_merged_format_right.format(msg=line_1))
	print(line_merged_format_right.format(msg=line_2))

	if new_tile is not None:
		print(line_merged_format_right.format(msg="%d: %s  " % (i, new_tile.symbol)))
	print(horizontal_line)


def generate_TG_boad(player_name, fixed_hand, hand, neighbors, game, new_tile=None, print_stolen_tiles=False):
	line_format_left = u"|{msg:<25s}|\n"
	line_format_right = u"|{msg:>25s}|\n"
	horizontal_line = line_format_left.format(msg='-' * 25)

	result = line_format_left.format(msg="Game of %s wind [%d]" % (game.game_wind, game.deck_size))

	for i in range(len(neighbors)):
		neighbor = neighbors[i]
		identifier = "%s" % neighbor.name
		if i == 0:
			identifier += " (next)"
		elif i == 2:
			identifier += " (prev)"

		fixed_hand_strs = []
		for meld_type, is_secret, tiles in neighbor.fixed_hand:
			meld_str = ""
			if is_secret:
				meld_str += Tile.tile_back_symbol + tiles[0].symbol + tiles[0].symbol + Tile.tile_back_symbol
			else:
				meld_str += "".join([tile.symbol for tile in tiles])
			fixed_hand_strs.append(meld_str)

		result += line_format_left.format(msg=identifier)
		result += line_format_left.format(msg=" ".join(fixed_hand_strs))
		result += line_format_right.format(
			msg="%s [%d]" % (Tile.tile_back_symbol * neighbor.hand_size, neighbor.hand_size))
		result += horizontal_line

	result += line_format_left.format(msg="Tiles disposed")
	disposed_tiles = game.disposed_tiles
	while True:
		result += line_format_left.format(msg="".join([tile.symbol for tile in disposed_tiles[0:25]]))
		disposed_tiles = disposed_tiles[25:]
		if len(disposed_tiles) == 0:
			break

	result += horizontal_line

	fixed_hand_strs, hand_str = [], ""
	for meld_type, is_secret, tiles in fixed_hand:
		meld_str = ""
		if is_secret:
			meld_str += Tile.tile_back_symbol + tiles[0].symbol + tiles[0].symbol + Tile.tile_back_symbol
		else:
			meld_str += "".join([tile.symbol for tile in tiles])
		fixed_hand_strs.append(meld_str)

	for tile in hand:
		hand_str += tile.symbol
	if new_tile is not None:
		hand_str += " - " + new_tile.symbol + " "

	result += line_format_left.format(msg="Your tiles")
	result += line_format_left.format(msg=" ".join(fixed_hand_strs))
	result += line_format_right.format(msg=hand_str)

	print(result)
	return result


def makesure_dir_exists(path):
	try:
		os.makedirs(path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise


def handpredictor_preprocessing(raw_data, hand_matrix_format):
	hand_matrix_format_choices = ["distrib", "exist", "raw_count"]

	if hand_matrix_format not in hand_matrix_format_choices:
		raise Exception("hand_matrix_format must be one of %s" % hand_matrix_format_choices)

	# n_data = raw_data["disposed_tiles_matrix"].shape[0]*4
	n_data = raw_data["disposed_tiles_matrix"].shape[0]
	processed_X = np.zeros((n_data, 4, 9, 4))
	processed_y = np.zeros((n_data, 34))

	common_disposed = raw_data["disposed_tiles_matrix"].sum(axis=1) / 4.0
	common_disposed = np.lib.pad(common_disposed, ((0, 0), (0, 2)), mode="constant", constant_values=0).reshape(
		(-1, 4, 9))

	common_fixed_hand = raw_data["fixed_hand_matrix"].sum(axis=1) / 4.0
	common_fixed_hand = np.lib.pad(common_fixed_hand, ((0, 0), (0, 2)), mode="constant", constant_values=0).reshape(
		(-1, 4, 9))

	raw_data["disposed_tiles_matrix"] = raw_data["disposed_tiles_matrix"].reshape([-1, 34]) / 4.0
	raw_data["disposed_tiles_matrix"] = np.lib.pad(raw_data["disposed_tiles_matrix"], ((0, 0), (0, 2)), mode="constant",
												   constant_values=0).reshape([-1, 4, 4, 9])

	raw_data["fixed_hand_matrix"] = raw_data["fixed_hand_matrix"].reshape([-1, 34]) / 4.0
	raw_data["fixed_hand_matrix"] = np.lib.pad(raw_data["fixed_hand_matrix"], ((0, 0), (0, 2)), mode="constant",
											   constant_values=0).reshape([-1, 4, 4, 9])

	if hand_matrix_format == "exist":
		raw_data["hand_matrix"] = np.greater(raw_data["hand_matrix"], 0) * 1.0
	elif hand_matrix_format == "distrib":
		raw_data["hand_matrix"] = normalize(raw_data["hand_matrix"].reshape([-1, 34]), norm="l1", axis=1).reshape(
			[-1, 4, 34])

	for i in range(raw_data["disposed_tiles_matrix"].shape[0]):
		'''
		processed_X[i*4:(i+1)*4, :, :, 0] = common_disposed[i, :, :]
		processed_X[i*4:(i+1)*4, :, :, 1] = raw_data["disposed_tiles_matrix"][i, :, :, :]
		processed_X[i*4:(i+1)*4, :, :, 2] = raw_data["fixed_hand_matrix"][i, :, :, :]
		processed_X[i*4:(i+1)*4, :, :, 3] = common_fixed_hand[i, :, :]
		processed_y[i*4:(i+1)*4, :] = raw_data["hand_matrix"][i, :, :]
		'''
		j = random.choice(range(4))
		processed_X[i, :, :, 0] = common_disposed[i, :, :]
		processed_X[i, :, :, 1] = raw_data["disposed_tiles_matrix"][i, j, :, :]
		processed_X[i, :, :, 2] = raw_data["fixed_hand_matrix"][i, j, :, :]
		processed_X[i, :, :, 3] = common_fixed_hand[i, :, :]
		processed_y[i, :] = raw_data["hand_matrix"][i, j, :]

	return processed_X, processed_y
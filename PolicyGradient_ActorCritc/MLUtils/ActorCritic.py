import tensorflow as tf
import numpy as np
from . import utils
import json
import warnings

save_file_name = "savefile.ckpt"
parameters_file_name = "paras.json"
loaded_models = {
	
}

n_actions = 42
sample_shape = [9, 34, 1]
sample_n_inputs = 9 * 34 * 1


def get_ActorCritic(path, **kwargs):
	if path not in loaded_models:
		try:
			print("Load Model")
			loaded_models[path] = [Actor.load(path),Critic.load(path)]
		except Exception as e:
			print(e)
			loaded_models[path] = [Actor(**kwargs),Critic(**kwargs)]
	return loaded_models[path]

class Actor(object):
	def __init__(self, from_save = None, learning_rate = 0.001, reward_decay = 0.95):
		self.__ep_obs, self.__ep_as, self.__ep_rs, self.__ep_a_filter = [], [], [], []
		tf.reset_default_graph()
		self.__graph = tf.Graph()
		self.__config = tf.ConfigProto(**utils.parallel_parameters)
		self.__sess = tf.Session(graph = self.__graph, config = self.__config)
		self.__invalide = 0.5
		self.__learn_step_counter = 0
		self.__average_loss = 0
		
		with self.__graph.as_default() as g:
			if from_save is None:
				print("New Network")
				self.__build_graph(None, learning_rate)
				self.__reward_decay = reward_decay
				self.__sess.run(tf.global_variables_initializer())
				self.__learn_step_counter = 0
				#self.__is_deep = None
			else:
				#read the model
				raise Exception("pass reading")
				#print("pass reading")
				pass
	def __build_graph(self, is_deep, learning_rate):
		w_init, b_init = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
		with tf.name_scope('inputs'):
			self.__obs = tf.placeholder(tf.float32, [None] + sample_shape, name = "observations")
			self.__acts = tf.placeholder(tf.int32, [None,], name = "actions_num")
			self.__vt = tf.placeholder(tf.float32, [None,], name = "actions_value")
			self.__a_filter = tf.placeholder(tf.float32, [None, n_actions], name = "actions_filter")
		
		state = tf.reshape(self.__obs,[-1, 9*34])
		
		dense_1 = tf.layers.dense(inputs = state, units = 3072, activation = tf.nn.sigmoid,kernel_initializer=w_init,bias_initializer=b_init)
		dense_2 = tf.layers.dense(inputs = dense_1, units = 1024, activation = tf.nn.sigmoid,kernel_initializer=w_init,bias_initializer=b_init)
		dense_3 = tf.layers.dense(inputs = dense_2, units = n_actions, activation = None,kernel_initializer=w_init,bias_initializer=b_init)
		
		a_dense_1 = tf.layers.dense(inputs = self.__a_filter, units = n_actions, activation = tf.nn.sigmoid,kernel_initializer=w_init,bias_initializer=b_init)
		a_dense_2 = tf.layers.dense(inputs = a_dense_1, units = n_actions, activation = None,kernel_initializer=w_init,bias_initializer=b_init)
		
		result = tf.add(dense_3,a_dense_2)
		
		self.__all_act_prob = tf.nn.softmax(result)
		
		with tf.name_scope('loss'):
			neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = result, labels = self.__acts)
			
			self.__loss = tf.reduce_mean(neg_log_prob * self.__vt)
		
		with tf.name_scope('train'):
			self.__train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.__loss)
		
		print("finished building graph")
	
	def choose_action(self, observation, action_filter = None, return_value = False, strict_filter = True):
		
		if action_filter is None:
			action_filter = np.full(n_actions, 1.0)
			
		prob_weights = self.__sess.run(
			self.__all_act_prob, 
			feed_dict = {
				self.__obs: observation[np.newaxis, :],
				self.__a_filter: action_filter[np.newaxis, :]
			})
		
		if strict_filter:
			valid_actions = np.where(action_filter > 0)[0]
			prob_weights_reduced = prob_weights.ravel()[valid_actions]
			#If the weights are too small, give equal proba
			#Else reduce and rearrange
			if prob_weights_reduced.sum() < 1e-5:
				prob_weights_reduced = np.full(prob_weights_reduced.shape[0], 1.0/prob_weights_reduced.shape[0])
			else:
				prob_weights_reduced = prob_weights_reduced / prob_weights_reduced.sum()
			action = np.random.choice(valid_actions.tolist(), p = prob_weights_reduced)
		else:
			action = np.random.choice(range(prob_weights.shape[1]), p = prob_weights.ravel())
		
		value = prob_weights[:,action]
		
		if return_value:
			return action,value
		
		return action
	
	def store_transition(self, state, action, reward, a_filter):
		self.__ep_obs.append(state)
		self.__ep_as.append(action)
		self.__ep_rs.append(reward)
		self.__ep_a_filter.append(a_filter)
		
	
	def learn(self, obs,action,reward,a_filter,display_cost = True):
		

		
		loss = np.NAN
		
		_, loss = self.__sess.run(
			[self.__train_op, self.__loss], 
			feed_dict={
				self.__obs: [obs],
				self.__acts: [action],
				self.__vt: [reward],
				self.__a_filter: [a_filter]
			}
		)
		
		self.__average_loss = 0.99 * self.__average_loss + 0.01 * loss
		
		if display_cost and self.__learn_step_counter%100==0:
			print("#%4d: %.4f"%(self.__learn_step_counter + 1, self.__average_loss))
		self.__learn_step_counter += 1

		return loss
	
	def save(self, save_dir):
		paras_dict = {
			"__reward_decay": self.__reward_decay,
			"__learn_step_counter": self.__learn_step_counter,
			"__is_deep": self.__is_deep
		}
		with open(save_dir.rstrip("/") + "/" + parameters_file_name, "w") as f:
			json.dump(paras_dict, f, indent = 4)

		with self.__graph.as_default() as g:
			saver = tf.train.Saver()
			save_path = saver.save(self.__sess, save_path = save_dir.rstrip("/")+"/"+save_file_name)
		tf.reset_default_graph()

	@classmethod
	def load(cls, path):
		model = cls(from_save = path)
		return model
	
class Critic(object):
	def __init__(self, from_save = None, learning_rate = 0.01, reward_decay = 0.9, gamma = 0.9):
		self.__n_actions = n_actions
		self.__graph = tf.Graph()
		self.__config = tf.ConfigProto(**utils.parallel_parameters)
		
		self.__sess = tf.Session(graph = self.__graph, config = self.__config)

		with self.__graph.as_default() as g:
			if from_save is None:
				self.__reward_decay = reward_decay
				self.__gamma = gamma
				self.__build_graph(None, learning_rate)
				self.__sess.run(tf.global_variables_initializer())
			else:
				#read the model
				raise Exception("pass reading")
				#print("pass reading")
				pass
	
	def __build_graph(self,is_deep,learning_rate):
		w_init, b_init = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
		
		with tf.name_scope('inputs'):
			self.__s = tf.placeholder(tf.float32, [None] + sample_shape, name = "s")
			self.__s_ = tf.placeholder(tf.float32, [None] + sample_shape, name = "s_")
			self.__r = tf.placeholder(tf.float32, [None, ], name = "r")
			self.__a_filter = tf.placeholder(tf.float32, [None, n_actions], name = "a_filter")
			self.__v_ = tf.placeholder(tf.float32, [None] + [1], name = "v_")
		
		
		state = tf.reshape(self.__s, [-1, 9*34])
		
		dense_1 = tf.layers.dense(inputs = state, units = 3072, activation = tf.nn.sigmoid,kernel_initializer=w_init,bias_initializer=b_init)
		dense_2 = tf.layers.dense(inputs = dense_1, units = 1024, activation = tf.nn.sigmoid,kernel_initializer=w_init,bias_initializer=b_init)
		self.__v = tf.layers.dense(inputs = dense_2, units = 1, activation = None,kernel_initializer=w_init,bias_initializer=b_init)
		
		with tf.variable_scope('squared_TD_error'):
			self.__td_error = self.__r + self.__gamma * self.__v_ - self.__v
			self.loss = tf.square(self.__td_error)
		
		with tf.variable_scope('train'):
			self.__train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
		
	def learn(self, s, r , s_):
		
		s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
		
		v_ = self.__sess.run(self.__v, {self.__s: s_})
		
		td_error, _ = self.__sess.run([self.__td_error, self.__train_op],
{self.__s: s, self.__v_: v_, self.__r: [r]})
		
		return td_error
		
	def save(self, save_dir):

		with self.__graph.as_default() as g:
			saver = tf.train.Saver()
			save_path = saver.save(self.__sess, save_path = save_dir.rstrip("/")+"/"+save_file_name)
		tf.reset_default_graph()

	@classmethod
	def load(cls, path):
		model = cls(from_save = path)
		return model
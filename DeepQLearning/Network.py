import utils
import tensorflow as tf
import random
import numpy as np
import json

save_file_name = "model.ckpt"
parameters_file_name = "parameters.json"
gpu_usage_w_limit = True
loaded_models = {}

n_actions = 42
sample_shape = [9, 34, 1]
sample_n_inputs = 9 * 34

def get_Network(path, **kwargs):
    if path not in loaded_models:
        try:
            loaded_models[path] = Network.load(path)
        except Exception as e:
            print(e)
            loaded_models[path] = Network(**kwargs)

    return loaded_models[path]

class Network:

    def __init__(self, from_save = None, learning_rate = 1e-2, reward_decay = 0.9, e_greedy = 0.9, replace_target_iter = 300, memory_size = 500, batch_size = 100):
        self._graph = tf.Graph()
        self._config = tf.ConfigProto(**utils.parallel_parameters)
        self._sess = tf.Session(graph=self._graph, config=self._config)
        self._memory_size = memory_size
        self._epsilon = e_greedy
        self._replace_tar_iter = replace_target_iter
        self._batch_size = batch_size
        self._learn_step = 0

        with self._graph.as_default() as g:
            if from_save is None:
                self.build_graph(learning_rate, reward_decay)
                self._sess.run(tf.global_variables_initializer())
            else:
                with open(from_save.rstrip("/") + "/" + parameters_file_name, "r") as f:
                    paras_dict = json.load(f)
                for key, value in paras_dict.items():
                    self.__dict__["_%s%s" % (self.__class__.__name__, key)] = value

                saver = tf.train.import_meta_graph(from_save.rstrip("/") + "/" + save_file_name + ".meta")
                saver.restore(self._sess, from_save.rstrip("/") + "/" + save_file_name)
                self._s = g.get_tensor_by_name("s:0")
                self._s_ = g.get_tensor_by_name("s_:0")
                self._r = g.get_tensor_by_name("r:0")
                self._a = g.get_tensor_by_name("a:0")
                self._a_filter = g.get_tensor_by_name("a_filter:0")
                self._is_train = g.get_tensor_by_name("is_train:0")
                self._q_eval = tf.get_collection("q_eval")[0]
                self._loss = tf.get_collection("loss")[0]
                self._train_op = tf.get_collection("train_op")[0]
                self._target_replace_op = tf.get_collection("target_replace_op")

        self._memory_counter = 0
        self._memory = np.zeros((self._memory_size, sample_n_inputs * 2 + 2 + n_actions))

        tf.reset_default_graph()

    def build_graph(self, learning_rate, reward_decay):
        w_init, b_init = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        self._s = tf.placeholder(tf.float32, [None] + sample_shape, name="s")
        self._s_ = tf.placeholder(tf.float32, [None] + sample_shape, name="s_")
        self._r = tf.placeholder(tf.float32, [None, ], name="r")
        self._a = tf.placeholder(tf.int32, [None, ], name="a")
        self._a_filter = tf.placeholder(tf.float32, [None, n_actions], name="a_filter")
        self._is_train = tf.placeholder(tf.bool, [], name="is_train")

        def connect(state, action_filter, name):
            collects = [name, tf.GraphKeys.GLOBAL_VARIABLES]
            hand_negated = tf.multiply(state[:, 0:1, :, :], tf.constant(-1.0))
            chows = tf.multiply(hand_negated, tf.constant(-1.0))
            tile_used = tf.reduce_sum(state[:, 1:, :, :], axis=1, keepdims=True)
            input_all = tf.concat([state[:, 0:2, :, :], chows, tile_used], axis=1)
            input_flat = tf.reshape(input_all, [-1, 4 * 34])

            weight_1 = tf.get_variable("weight_1", [4 * 34, 3072], initializer=w_init, collections=collects)
            bias_1 = tf.get_variable("bias_1", [3072], initializer=b_init, collections=collects)
            layer_1 = tf.sigmoid(tf.matmul(input_flat, weight_1) + bias_1)

            weight_2 = tf.get_variable("weight_2", [3072, 1024], initializer=w_init, collections=collects)
            bias_2 = tf.get_variable("bias_2", [1024], initializer=b_init, collections=collects)
            layer_2 = tf.sigmoid(tf.matmul(layer_1, weight_2) + bias_2)

            weight_3 = tf.get_variable("weight_3", [1024, n_actions], initializer=w_init, collections=collects)
            bias_3 = tf.get_variable("bias_3", [n_actions], initializer=b_init, collections=collects)

            return tf.multiply(tf.matmul(layer_2, weight_3) + bias_3, action_filter)

        with tf.variable_scope("q_eval_net"):
            self._q_eval = connect(self._s, self._a_filter, "q_eval_net_params")

        with tf.variable_scope("target_net"):
            self._q_next = connect(self._s_, self._a_filter, "target_net_params")

        self._q_target = tf.stop_gradient(self._r + reward_decay * tf.reduce_max(self._q_next, axis=1))

        a_indices = tf.stack([tf.range(tf.shape(self._a)[0], dtype=tf.int32), self._a], axis=1)
        self._q_eval_wrt_a = tf.gather_nd(params=self._q_eval, indices=a_indices)

        self._loss = tf.reduce_mean(tf.squared_difference(self._q_target, self._q_eval_wrt_a), name="TD_error")
        self._train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(self._loss)

        q_eval_net_params = tf.get_collection("q_eval_net_params")
        target_net_params = tf.get_collection("target_net_params")
        self._target_replace_op = [tf.assign(t, e) for t, e in zip(target_net_params, q_eval_net_params)]

        tf.add_to_collection("q_eval", self._q_eval)
        tf.add_to_collection("loss", self._loss)
        tf.add_to_collection("train_op", self._train_op)
        for op in self._target_replace_op:
            tf.add_to_collection("target_replace_op", op)

    def save(self, save_dir):
        paras_dict = {
            "_epsilon": self._epsilon,
            "_memory_size": self._memory_size,
            "_replace_target_iter": self._replace_tar_iter,
            "_batch_size": self._batch_size,
            "_learn_step": self._learn_step,
        }
        with open(save_dir.rstrip("/") + "/" + parameters_file_name, "w") as f:
            json.dump(paras_dict, f, indent=4)

        with self._graph.as_default() as g:
            saver = tf.train.Saver()
            save_path = saver.save(self._sess, save_path=save_dir.rstrip("/") + "/" + save_file_name)
        tf.reset_default_graph()

    @classmethod
    def load(cls, path):
        model = cls(from_save=path)
        return model

    def store_transition(self, state, action, reward, state_, action_filter=None):
        if action_filter is None:
            action_filter = np.full(n_actions, 1)

        transition = np.hstack(
            (state.reshape(sample_n_inputs), [action, reward], state_.reshape(sample_n_inputs), action_filter))
        index = self._memory_counter % self._memory_size
        self._memory[index, :] = transition
        self._memory_counter += 1

    def choose_action(self, state, action_filter=None, eps_greedy=True, return_value=False, strict_filter=False):
        if action_filter is None:
            action_filter = np.full(n_actions, 1)

        if np.random.uniform() < self._epsilon or not eps_greedy:
            inputs = state[np.newaxis, :]
            action_filter = action_filter[np.newaxis, :]

            with self._graph.as_default() as g:
                actions_value = self._sess.run(self._q_eval,
                                                feed_dict={self._s: inputs, self._a_filter: action_filter,
                                                           self._is_train: False})

            tf.reset_default_graph()
            if strict_filter:
                valid_actions = np.where(action_filter[0, :] > 0)[0]
                action = valid_actions[np.argmax(actions_value[0, valid_actions])]
            else:
                action = np.argmax(actions_value)
            value = actions_value[0, action]
        else:
            action = random.choice(np.arange(n_actions)[action_filter >= 0])
            value = np.nan

        if return_value:
            return action, value

        return action

    def learn(self, display_cost=True):
        if self._learn_step % self._replace_tar_iter == 0:
            self._sess.run(self._target_replace_op)
        cost = None
        sample_index = np.random.choice(min(self._memory_size, self._memory_counter), size=self._batch_size)
        batch_memory = self._memory[sample_index, :]
        with self._graph.as_default() as g:
            _, cost = self._sess.run(
                [self._train_op, self._loss],
                feed_dict={
                    self._s: batch_memory[:, :sample_n_inputs].reshape([-1] + sample_shape),
                    self._a: batch_memory[:, sample_n_inputs],
                    self._r: batch_memory[:, sample_n_inputs + 1],
                    self._s_: batch_memory[:, (sample_n_inputs + 2):(sample_n_inputs * 2 + 2)].reshape(
                        [-1] + sample_shape),
                    self._a_filter: batch_memory[:, (sample_n_inputs * 2 + 2):],
                    self._is_train: True
                })
        tf.reset_default_graph()
        if display_cost:
            print("#%4d: %.4f" % (self._learn_step + 1, cost))
        self._learn_step += 1
        return cost

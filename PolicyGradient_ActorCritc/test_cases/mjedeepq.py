'''
This experiment trains the extended deep Q model by letting it play against heuristics based players and players that take random moves.
The model is extended as the action set now includes different kinds of Chow and Pong, like Chow on Bamboo tiles, Chow on Dots tiles, etc
The model could also train the deepqpr model, which stands for Deep Q with prioritized experience replay.
'''
import argparse
import numpy as np
import random
import signal
from MLUtils import get_MJEDeepQNetwork, get_MJEDeepQNetworkPR, get_MJEDeepQNetworkPRD
import Player, Game
import MoveGenerator
from . import utils

EXIT_FLAG = False
network_type = "pr"
model_flag = "deepq"
names = ["Amy", "Billy", "Clark", "David"]
freq_shuffle_players = 8
freq_model_save = None
game_record_size = 100
game_record_count = 0

# heuristics_deepq, heuristics
game_record = np.zeros((game_record_size, 4, 2))

deep_q_model_paras = {
	"is_deep": False,
	"learning_rate": 1e-3,
	"reward_decay": 0.9, 
	"e_greedy": 0.8,
	"replace_target_iter": 300, 
	"memory_size": 1000, 
	"batch_size": 300,
	"n_actions": 48 if model_flag == "deepq" else 39
}
deep_q_model_dir = "rule_base_q_test"

trainer_conf = [model_flag, model_flag, model_flag]
trainer_pool = ["heuristics", "heuristics2", "heuristics3", "random", model_flag]

trainer_models = {
	"heuristics": {
		"class": MoveGenerator.RuleBasedAINaive,
		"parameters":{
			 "display_step": False,
			 "s_chow": 2,
			 "s_pong": 6,
			 "s_future": 1,
			 "s_explore": 0,
			 "s_neighbor_suit": 0,
			 "s_mixed_suit": 0
		}
	},
	"heuristics2": {
		"class": MoveGenerator.RuleBasedAINaive,
		"parameters":{
			 "display_step": False,
			 "s_chow": 2,
			 "s_pong": 6,
			 "s_future": 1,
			 "s_explore": 0.35,
			 "s_neighbor_suit": 0,
			 "s_mixed_suit": 0
		}
	},
	"heuristics3": {
		"class": MoveGenerator.RuleBasedAINaive,
		"parameters":{
			 "display_step": False,
			 "s_chow": 2,
			 "s_pong": 6,
			 "s_future": 1,
			 "s_explore": 0.7,
			 "s_neighbor_suit": 0,
			 "s_mixed_suit": 0
		}
	},
	"deepq": {
		"class": MoveGenerator.DeepQEGenerator,
		"parameters": {
			"network_type": network_type,
			"display_step": False,
			"q_network_path": deep_q_model_dir,
			"is_train": False,
			"skip_history": False
		}
	},
	"deepqr":{
		"class": MoveGenerator.DeepQRGenerator,
		"parameters": {
			"network_type": network_type,
			"display_step": False,
			"q_network_path": deep_q_model_dir,
			"is_train": False,
			"skip_history": False
		}
	},
	"random": {
		"class": MoveGenerator.RandomGenerator,
		"parameters":{
			"display_step": False
		}
	}
}
edeepq_getter_map = {
	"vanilla": get_MJEDeepQNetwork,
	"pr": get_MJEDeepQNetworkPR,
	"prd": get_MJEDeepQNetworkPRD
}
get_network = edeepq_getter_map[network_type]

def signal_handler(signal, frame):
	global EXIT_FLAG
	print("Signal received, cleaning up..")
	EXIT_FLAG = True

def parse_args(args_list):
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_dir", type = str, help = "Where is the model")
	parser.add_argument("--rand_opp", action = "store_true", default = False, help = "randomly choose opponents")
	parser.add_argument("action", type = str, choices = ["train", "test", "play"], help = "What to do with the model")
	parser.add_argument("n_episodes", nargs = "?", default = 1, type = int, help = "No. of episodes to go through")
	parser.add_argument("save_name", nargs = "?", default = None, type = str, help = "Path to save the model")
	args = parser.parse_args(args_list)
	return args

def get_oppponents(rand_opp, all_human):
	players = []
	for i in range(3):
		if all_human:
			players.append(Player.Player(MoveGenerator.Human, player_name = names[i]))
		else:
			model_tag = random.choice(trainer_pool) if rand_opp else trainer_conf[i]
			players.append(Player.Player(trainer_models[model_tag]["class"], player_name = names[i], **trainer_models[model_tag]["parameters"]))
	return players

def test(args):
	global game_record_count
	args = parse_args(args)

	if args.action == "train":
		if args.save_name is None:
			response = input("You have not entered save_name, are you sure? [y/n] ").lower()
			if response != "y":
				exit(-1)

		if args.model_dir is None:
			args.model_dir = deep_q_model_dir
		else:
			trainer_models["deepq"]["parameters"]["q_network_path"] = args.model_dir
			trainer_models["deepqr"]["parameters"]["q_network_path"] = args.model_dir
			
		freq_model_save = args.n_episodes//10

	elif args.action in ["test", "play"]:
		if args.model_dir is None:
			raise Exception("model_dir must be given to test/play")

	model = get_network(args.model_dir, **deep_q_model_paras)
	deepq_player = Player.Player(MoveGenerator.DeepQEGenerator if model_flag == "deepq" else MoveGenerator.DeepQRGenerator, player_name = names[len(names) - 1], q_network_path = args.model_dir, network_type = network_type, skip_history = False, is_train = args.action == "train", display_step = args.action == "play")

	if args.action != "play":
		signal.signal(signal.SIGINT, signal_handler)
	game, shuffled_players, last_saved = None, None, -1
	for i in range(args.n_episodes):
		if EXIT_FLAG:
			break

		if i % freq_shuffle_players == 0:
			if args.rand_opp or i == 0:
				players = get_oppponents(args.rand_opp, args.action == "play")
				players.append(deepq_player)

			shuffled_players = random.sample(players, k = 4)
			game = Game.Game(shuffled_players)

		winner, losers, penalty = game.start_game()
		if args.action == "train":
			model.learn(display_cost = (i+1) % game_record_size == 0)
		
		index = game_record_count%game_record_size
		game_record[index, :, :] = np.zeros((4, 2))
		game_record_count += 1

		if winner is not None:
			winner_id = players.index(winner)
			game_record[index, winner_id, 0] = 1
			for loser in losers:
				loser_id = players.index(loser)
				game_record[index, loser_id, 1] = 1

		if (i+1) % game_record_size == 0:
			print("#%5d: %.2f%%/%.2f%%\t%.2f%%/%.2f%%\t%.2f%%/%.2f%%\t%.2f%%/%.2f%%"%(i+1, game_record[:, 0, 0].mean()* 100, game_record[:, 0, 1].mean()* 100,
																							game_record[:, 1, 0].mean()* 100, game_record[:, 1, 1].mean()* 100, 
																							game_record[:, 2, 0].mean()* 100, game_record[:, 2, 1].mean()* 100, 
																							game_record[:, 3, 0].mean()* 100, game_record[:, 3, 1].mean()* 100))
		'''
		if args.action == "train" and args.save_name is not None and (i+1) % freq_model_save == 0:
			last_saved = i
			path = args.save_name.rstrip("/") + "_%d"%(i + 1)
			utils.makesure_dir_exists(path)
			model.save(path)
		'''

	if args.action == "train" and args.save_name is not None:
		if last_saved < args.n_episodes - 1:
			path = args.save_name.rstrip("/") + "_%d"%args.n_episodes
			utils.makesure_dir_exists(path)
			model.save(path)

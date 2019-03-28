from DeepQLearning import utils
import random
import numpy as np
import Tile
import MoveGenerator
from TGLanguage import get_text
from DeepQLearning.Network import get_Network
import argparse
import signal
import Player, Game

display_name = "DeepQ"
save_name = "DeepQ"

REWARD_VICTORY = 200
REWARD_DRAW = -5
REWARD_LOSE = -100
REWARD_INVALID_DECISION = -5
REWARD_NON_TERMINAL = 0
n_epochs = 3000


n_decisions = 42
decisions_ = ["dots_chow", "dots_pong", "characters_chow", "characters_pong", "bamboo_chow", "bamboo_pong", "honor_pong", "no_action"]

EXIT_FLAG = False
names = ["Amy", "Billy", "Clark", "David"]
freq_shuffle_players = 8
freq_model_save = None
game_record_size = 100
game_record_count = 0


game_record = np.zeros((game_record_size, 4, 2))

deep_q_model_paras = {
    "learning_rate": 1e-3,
    "reward_decay": 0.9,
    "e_greedy": 0.8,
    "replace_target_iter": 300,
    "memory_size": 1000,
    "batch_size": 300
}
deep_q_model_dir = "DeepQ_10"




def signal_handler(signal, frame):
    global EXIT_FLAG
    print("Signal received, cleaning up..")
    EXIT_FLAG = True

def parse_args(args_list):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type = str, help = "Where is the model")
    parser.add_argument("action", type = str, choices = ["train", "test", "play"], help = "What to do with the model")
    parser.add_argument("n_episodes", nargs = "?", default = 1, type = int, help = "No. of episodes to go through")
    parser.add_argument("save_name", nargs = "?", default = None, type = str, help = "Path to save the model")
    args = parser.parse_args(args_list)
    return args


class Generator(MoveGenerator.MoveGenerator):

    def __init__(self, player_name, q_network_path, is_train, skip_history=False, display_tgboard=False,
                 display_step=False):
        super(Generator, self).__init__(player_name, display_tgboard=display_tgboard)
        self.display_step = display_step
        self.q_network_path = q_network_path
        self.is_train = is_train
        self.skip_history = skip_history
        self.clear_history()
        self.history_waiting = None
        self.q_network_history = {}

    def print_msg(self, msg):
        if self.display_step:
            print(msg)

    def reset_new_game(self):
        if self.is_train and self.history_waiting:
            self.update_transition("terminal", REWARD_DRAW)
            self.history_waiting = False

    def notify_loss(self, score):
        if self.is_train and self.history_waiting:
            self.update_transition("terminal", REWARD_LOSE)
            self.history_waiting = False

    def update_history(self, state, action, action_filter):
        if not self.is_train:
            return

        if self.history_waiting:
            raise Exception("the network is waiting for a transition")

        self.history_waiting = True
        self.q_network_history["state"] = state
        self.q_network_history["action"] = action
        self.q_network_history["action_filter"] = action_filter

    def update_transition(self, state_, reward=0, action_filter_=None):
        if not self.is_train:
            return

        if not self.history_waiting:
            raise Exception("the network is NOT waiting for a transition")

        if type(state_) == str and state_ == "terminal":
            state_ = self.q_network_history["state"]

        self.history_waiting = False
        q_network = get_Network(self.q_network_path)
        q_network.store_transition(self.q_network_history["state"], self.q_network_history["action"], reward, state_,
                                   self.q_network_history["action_filter"])

    def clear_history(self):
        self.history_waiting = False
        self.q_network_history = {
            "state": None,
            "action": None,
            "action_filter": None
        }

    def decide_chow(self, player, new_tile, choices, neighbors, game):
        self.begin_decision()

        fixed_hand, hand = player.fixed_hand, player.hand

        if self.display_step:
            self.print_game_board(fixed_hand, hand, neighbors, game)

        self.print_msg("Someone just discarded a %s." % new_tile.symbol)

        q_network = get_Network(self.q_network_path)
        state = utils.dnn_encode_state(player, neighbors)

# store the transition to the network (state(t-1), action(t-1), reward(not terminated), state(t)
        if not self.skip_history and self.history_waiting:
            self.update_transition(state, REWARD_NON_TERMINAL)

        valid_actions = [34 + decisions_.index("%s_chow" % new_tile.suit), 34 + decisions_.index("no_action")]
        action_filter = np.zeros(n_decisions)
        action_filter[valid_actions] = 1
        action = None

# choose the action
        while True:
            if action is not None and not self.skip_history:
                self.update_history(state, action, action_filter)
                self.update_transition(state, REWARD_INVALID_DECISION)

            action, value = q_network.choose_action(state, action_filter=action_filter, eps_greedy=self.is_train,
                                                    return_value=True, strict_filter=not self.is_train)

            if action in valid_actions:
                break
            elif not self.is_train:
                action = random.choice(valid_actions)
                break

        if not self.skip_history:
            self.update_history(state, action, action_filter)

        self.end_decision()

# print the msg of the choice taken
        if action == 34 + decisions_.index("no_action"):
            self.print_msg("%s chooses not to Chow %s [%.2f]." % (self.player_name, new_tile.symbol, value))
            return False, None
        else:
            chow_tiles_tgstrs = []
            chow_tiles_str = ""
            choice = random.choice(choices)
            for i in range(choice - 1, choice + 2):
                neighbor_tile = new_tile.generate_neighbor_tile(i)
                chow_tiles_str += neighbor_tile.symbol
                chow_tiles_tgstrs.append(neighbor_tile.get_display_name(game.lang_code, is_short=False))

            self.print_msg("%s chooses to Chow %s [%.2f]." % (self.player_name, chow_tiles_str, value))

            if game.lang_code is not None:
                game.add_notification(
                    get_text(game.lang_code, "NOTI_CHOOSE_CHOW") % (self.player_name, ",".join(chow_tiles_tgstrs)))

            return True, choice

# same as the previous method
    def decide_kong(self, player, new_tile, kong_tile, location, src, neighbors, game):
        self.begin_decision()
        fixed_hand, hand = player.fixed_hand, player.hand

        if self.display_step:
            self.print_game_board(fixed_hand, hand, neighbors, game, new_tile)

        if src == "steal":
            self.print_msg("Someone just discarded a %s." % kong_tile.symbol)
        elif src == "draw":
            self.print_msg("You just drew a %s" % kong_tile.symbol)
        elif src == "existing":
            self.print_msg("You have 4 %s in hand" % kong_tile.symbol)

        if location == "fixed_hand":
            location = "fixed hand"
        else:
            location = "hand"

        q_network = get_Network(self.q_network_path)
        state = utils.dnn_encode_state(player, neighbors)

        if not self.skip_history and self.history_waiting:
            self.update_transition(state, REWARD_NON_TERMINAL)

        valid_actions = [34 + decisions_.index("%s_pong" % new_tile.suit), 34 + decisions_.index("no_action")]
        action_filter = np.zeros(n_decisions)
        action_filter[valid_actions] = 1
        action = None

        while True:
            if action is not None and not self.skip_history:
                self.update_history(state, action, action_filter)
                self.update_transition(state, REWARD_INVALID_DECISION)

            action, value = q_network.choose_action(state, action_filter=action_filter, eps_greedy=self.is_train,
                                                    return_value=True, strict_filter=not self.is_train)

            if action in valid_actions:
                break
            elif not self.is_train:
                action = random.choice(valid_actions)
                break

        if not self.skip_history:
            self.update_history(state, action, action_filter)

        self.end_decision()

        if action == 34 + decisions_.index("no_action"):
            self.print_msg("%s [%s] chooses to form a Kong %s%s%s%s [%.2f]." % (
            self.player_name, display_name, kong_tile.symbol, kong_tile.symbol, kong_tile.symbol, kong_tile.symbol,value))
            if game.lang_code is not None:
                game.add_notification(get_text(game.lang_code, "NOTI_CHOOSE_KONG") % (
                self.player_name, kong_tile.get_display_name(game.lang_code, is_short=False)))

            return True
        else:
            self.print_msg("%s [%s] chooses not to form a Kong %s%s%s%s [%.2f]." % (
            self.player_name, display_name, kong_tile.symbol, kong_tile.symbol, kong_tile.symbol, kong_tile.symbol,value))
            return False

    def decide_pong(self, player, new_tile, neighbors, game):
        self.begin_decision()

        fixed_hand, hand = player.fixed_hand, player.hand

        if self.display_step:
            self.print_game_board(fixed_hand, hand, neighbors, game, new_tile)

        self.print_msg("Someone just discarded a %s." % new_tile.symbol)

        q_network = get_Network(self.q_network_path)
        state = utils.dnn_encode_state(player, neighbors)

        if not self.skip_history and self.history_waiting:
            self.update_transition(state, REWARD_NON_TERMINAL)

        valid_actions = [34 + decisions_.index("%s_pong" % new_tile.suit), 34 + decisions_.index("no_action")]
        action_filter = np.zeros(n_decisions)
        action_filter[valid_actions] = 1
        action = None

        while True:
            if action is not None and not self.skip_history:
                self.update_history(state, action, action_filter)
                self.update_transition(state, REWARD_INVALID_DECISION)

            action, value = q_network.choose_action(state, action_filter=action_filter, eps_greedy=self.is_train,
                                                    return_value=True, strict_filter=not self.is_train)

            if action in valid_actions:
                break
            elif not self.is_train:
                action = random.choice(valid_actions)
                break

        if not self.skip_history:
            self.update_history(state, action, action_filter)

        self.end_decision()
        if action == 34 + decisions_.index("no_action"):
            self.print_msg("%s [%s] chooses to form a Pong %s%s%s. [%.2f]" % (
            self.player_name, display_name, new_tile.symbol, new_tile.symbol, new_tile.symbol, value))
            if game.lang_code is not None:
                game.add_notification(get_text(game.lang_code, "NOTI_CHOOSE_PONG") % (
                self.player_name, new_tile.get_display_name(game.lang_code, is_short=False)))
            return True
        else:
            self.print_msg("%s [%s] chooses not to form a Pong %s%s%s. [%.2f]" % (
            self.player_name, display_name, new_tile.symbol, new_tile.symbol, new_tile.symbol, value))
            return False

    def decide_win(self, player, grouped_hand, new_tile, src, score, neighbors, game):
        self.begin_decision()
        if not self.skip_history and self.history_waiting:
            self.update_transition("terminal", REWARD_VICTORY)

        fixed_hand, hand = player.fixed_hand, player.hand
        if self.display_step:
            if src == "steal":
                self.print_game_board(fixed_hand, hand, neighbors, game)
                self.print_msg("Someone just discarded a %s." % new_tile.symbol)
            else:
                self.print_game_board(fixed_hand, hand, neighbors, game, new_tile=new_tile)

            self.print_msg("%s [%s] chooses to declare victory." % (self.player_name, display_name))
            if game.lang_code is not None:
                game.add_notification(get_text(game.lang_code, "NOTI_CHOOSE_VICT") % (self.player_name))

            self.print_msg("You can form a victory hand of: ")
            utils.print_hand(fixed_hand, end=" ")
            utils.print_hand(grouped_hand, end=" ")
            self.print_msg("[%d]" % score)

        self.end_decision()

        return True

    def decide_drop_tile(self, player, new_tile, neighbors, game):
        self.begin_decision()

        fixed_hand, hand = player.fixed_hand, player.hand
        state = utils.dnn_encode_state(player, neighbors)

        if not self.skip_history and self.history_waiting:
            self.update_transition(state, REWARD_NON_TERMINAL)

        if self.display_step:
            self.print_game_board(fixed_hand, hand, neighbors, game, new_tile)

        q_network = get_Network(self.q_network_path)

        valid_actions = []
        tiles = player.hand if new_tile is None else player.hand + [new_tile]
        for tile in tiles:
            valid_actions.append(Tile.convert_tile_index(tile))

        action_filter = np.zeros(n_decisions)
        action_filter[valid_actions] = 1
        action = None
        while True:
            if action is not None and not self.skip_history:
                self.update_history(state, action, action_filter)
                self.update_transition(state, REWARD_INVALID_DECISION)

            action, value = q_network.choose_action(state, action_filter=action_filter, eps_greedy=self.is_train,
                                                    return_value=True, strict_filter=not self.is_train)

            if action in valid_actions:
                break
            elif not self.is_train:
                action = random.choice(valid_actions)
                break

        if not self.skip_history:
            self.update_history(state, action, action_filter)
        drop_tile = Tile.convert_tile_index(action)
        self.print_msg("%s [%s] chooses to drop %s. [%.2f]" % (self.player_name, display_name, drop_tile.symbol, value))
        self.end_decision(True)

        if game.lang_code is not None:
            game.add_notification(get_text(game.lang_code, "NOTI_CHOOSE_DISCARD") % (
            self.player_name, drop_tile.get_display_name(game.lang_code, is_short=False)))

        return drop_tile

trainer_conf = ["random", "random", "random"]

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
    "deepq": {
        "class": Generator,
        "parameters": {
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


def main():
    global game_record_count
    trainer_models["deepq"]["parameters"]["q_network_path"] = deep_q_model_dir
    model = get_Network(deep_q_model_dir, **deep_q_model_paras)
    players = []
    i = 0
    for model_tag in trainer_conf:
        player = Player.Player(trainer_models[model_tag]["class"], player_name=names[i], **trainer_models[model_tag]["parameters"])
        players.append(player)
        i += 1
    deepq_player = Player.Player(Generator, player_name=names[i], q_network_path=deep_q_model_dir,
                                 skip_history=False, is_train=True,
                                 display_step=False)
    players.append(deepq_player)
    signal.signal(signal.SIGINT, signal_handler)
    game, shuffled_players, last_saved = None, None, -1
    for i in range(n_epochs):
        if EXIT_FLAG:
            break

        if i % freq_shuffle_players == 0:
            shuffled_players = random.sample(players, k=4)
            game = Game.Game(shuffled_players)

        winner, losers, penalty = game.start_game()
        model.learn(display_cost=(i + 1) % game_record_size == 0)

        index = game_record_count % game_record_size
        game_record[index, :, :] = np.zeros((4, 2))
        game_record_count += 1

        if winner is not None:
            winner_id = players.index(winner)
            game_record[index, winner_id, 0] = 1
            for loser in losers:
                loser_id = players.index(loser)
                game_record[index, loser_id, 1] = 1

        if (i + 1) % game_record_size == 0:
            print("#%5d: %.2f%%/%.2f%%\t%.2f%%/%.2f%%\t%.2f%%/%.2f%%\t%.2f%%/%.2f%%" % (
            i + 1, game_record[:, 0, 0].mean() * 100, game_record[:, 0, 1].mean() * 100,
            game_record[:, 1, 0].mean() * 100, game_record[:, 1, 1].mean() * 100,
            game_record[:, 2, 0].mean() * 100, game_record[:, 2, 1].mean() * 100,
            game_record[:, 3, 0].mean()* 100, game_record[:, 3, 1].mean()* 100))

        if last_saved < n_epochs - 1:
            path = save_name.rstrip("/") + "_%d" % n_epochs
            utils.makesure_dir_exists(path)
            model.save(path)


if __name__ == "__main__":
    main()
    print("finished")
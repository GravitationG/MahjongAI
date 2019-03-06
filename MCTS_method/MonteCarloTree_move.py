import random
import numpy as np
import Tile
import utils


##================================================================================================================================
# MCTS node class 
## Based on self.hand + self.fixed_hand + neighbors.fixed_hand + all_discarded tiles to take drop_tile decision
# At each tree node, we randomly draw a tile
##================================================================================================================================
class node():
    def __init__(self,
                 fixed_hand,
                 hand_map,
                 invisible_tiles_map,
                 invisible_tiles_number,
                 max_depth,
                 c=0.5,
                 parent=None,
                 depth=0,
                 action=None):
        self._fixed_hand = fixed_hand
        self._hand_map = hand_map
        self._invisible_tiles_map = invisible_tiles_map
        self._invisible_tiles_number = invisible_tiles_number

        self._visit_count = 0
        self._children = {}  # combination of (discard tile + new tile) pair
        self._possible_actions = {
        }  # action include: discard tile1, discard tile2, discard tile3 ..... end
        self._parent = parent
        self._c = c
        self._expand = False

        self._max_tile_action_score = 0
        self._max_child_score = 0
        self._avg_score = 0  # avg_simulated score
        self._depth = depth  # round_number
        self._max_depth = max_depth

        self._end = False
        self._action = action

    def update_value(self, visit, mj_score):
        if visit:
            self._visit_count += 1
        self._avg_score = (self._avg_score * (self._visit_count - 1) +
                           mj_score) / self._visit_count

    def update_possible_action(self, node):
        action = node._action
        try:
            avg_score = self._possible_actions[action]['avg_score']
            num_visit = self._possible_actions[action]['num_visit']
            self._possible_actions[action]['avg_score'] = (
                avg_score * num_visit + node._avg_score) / (num_visit + 1)
            self._possible_actions[action]['num_visit'] += 1
        except:
            pass

    def calculate_uct_tile_action(self):
        p = self._parent
        if p == None:
            total_visit = sum([
                info['num_visit']
                for _, info in self._possible_actions.items()
            ])
        else:
            total_visit = p._visit_count

        for tile, info in self._possible_actions.items():
            uct_score = 0
            if self._max_tile_action_score == 0:
                if info['num_visit'] != 0 and total_visit != 0:
                    uct_score = self._c * np.sqrt(
                        np.log(total_visit) / info['num_visit'])
            else:
                uct_score = info[
                    'avg_score'] / self._max_tile_action_score + self._c * np.sqrt(
                        np.log(total_visit) / info['num_visit'])
            info['uct_score'] = uct_score

    def max_avg_score_of_tile_action(self):
        l = []
        for tile, info in self._possible_actions.items():
            l.append(info['avg_score'])
        self._max_tile_action_score = max(l)

    def calculate_uct_child(self):
        p = self._parent
        if p == None:
            total_visit = sum(
                [chd._visit_count for _, chd in self._children.items()])
        else:
            total_visit = self._visit_count

        uct_child_scores = {}
        for key, child in self._children.items():
            if self._max_child_score == 0:
                if total_visit != 0 and child._visit_count != 0:
                    uct_child_scores[key] = self._c * np.sqrt(
                        np.log(total_visit) / child._visit_count)
                else:
                    uct_child_scores[key] = 0
            else:
                if total_visit != 0 and child._visit_count != 0:
                    uct_child_scores[
                        key] = child._avg_score / self._max_child_score + self._c * np.sqrt(
                            np.log(total_visit) / child._visit_count)
                else:
                    uct_child_scores[
                        key] = child._avg_score / self._max_child_score

        return uct_child_scores

    def max_avg_score_of_child(self):
        l = []
        for key, child in self._children.items():
            l.append(child._avg_score)
        self._max_child_score = max(l)

    def add_child(self, child, tag):
        self._children[tag] = child

    def expand_children(self):
        if len(self._children) > 0:
            return
        try:
            for discard_tile in self._hand_map:
                self._possible_actions[discard_tile] = {
                    'list_childs': [],
                    'avg_score': 0,
                    'num_visit': 0,
                    'uct_score': 0
                }
                # add child nodes
                for new_tile in self._invisible_tiles_map:
                    hand_map = self._hand_map.copy()
                    invisible_tiles_map = self._invisible_tiles_map.copy()
                    invisible_tiles_number = self._invisible_tiles_number
                    hand_map = utils.map_increment(
                        hand_map, discard_tile, -1, remove_zero=True)
                    invisible_tiles_map = utils.map_increment(
                        invisible_tiles_map, new_tile, -1, remove_zero=True)
                    invisible_tiles_number -= 1
                    hand_map = utils.map_increment(hand_map, new_tile, 1)
                    child = node(
                        self._fixed_hand,
                        hand_map,
                        invisible_tiles_map,
                        invisible_tiles_number,
                        self._max_depth,
                        depth=self._depth + 1,
                        parent=self,
                        action=discard_tile)
                    self._children[(discard_tile, new_tile)] = child
                    self._possible_actions[discard_tile]['list_childs'].append(
                        child)
            #self._possible_actions['end'] = {'list_childs': [], 'avg_score': 0, 'num_visit':0}
            self._expand = True
        except:
            pass


# search
def MCT_search(root, max_simulation=1000):
    order = 0
    #print("Begin search...")
    while order < max_simulation:
        action, leaf = MCT_traverse(root)
        if leaf != None:
            sim_result = MCT_rollout(leaf, action)
            MCT_backpropagate(root, leaf, sim_result)
        order += 1
    if root._possible_actions != {}:
        #root.max_avg_score_of_tile_action()
        # root.calculate_uct_tile_action()
        max_score = float('-inf')
        max_tile_action = None
        for tile, info in root._possible_actions.items():
            if info['avg_score'] > max_score:
                max_score = info['avg_score']
                max_tile_action = tile
        #print("End search....")
        if max_tile_action != None:
            return max_tile_action
        else:
            return random.sample(root._possible_actions.keys(), k=1)[0]
    else:
        #root.max_avg_score_of_child()
        max_s = float('-inf')
        max_c = None
        #uct_scores = root.calculate_uct_child()
        for key, child in root._children.items():
            if child._avg_score > max_s:
                max_s = child._avg_score
                max_c = key
        #print("End search....")
        #print(max_c)
        if max_c != None:
            if type(max_c) != tuple:
                return max_c
            else:
                return max_c[0]
        else:
            return random.sample(root._children.keys(), k=1)[0][0]

    #root
    #root.sort_children()
    return root.children[0]


# explore to the promising leaf
def MCT_traverse(root, max_depth=36):
    #print("Begin traverse")
    child_node = root
    tile_action = None
    while len(child_node._children) > 0 or child_node._visit_count > 0:
        tile_action, child_node = best_uct(child_node)
        if (type(tile_action) == str
                and tile_action == 'end') or child_node == None:
            break
    return tile_action, child_node


# simulation
def MCT_rollout(node, action):
    '''
    if type(action) == str and action =='end':
        # end node 
        if node._possible_actions['end']['num_visit']==0:
            score = map_hand_eval_func(node.fixed_hand, node._hand_map, node._invisible_tiles_map, node._invisible_tiles_number)
        else:
            score = node._possible_actions['end']['avg_score']
    else: 
        '''
    # simulation_node
    #print("Begin rollout")
    round_n = 0
    hand_map = node._hand_map.copy()
    invisible_tiles_map = node._invisible_tiles_map.copy()
    invisible_tiles_number = node._invisible_tiles_number

    while node._depth + round_n < node._max_depth and invisible_tiles_number > 0:
        discard_tile = random.sample(hand_map.keys(), 1)[0]
        new_tile = random.sample(invisible_tiles_map.keys(), 1)[0]

        hand_map = utils.map_increment(
            hand_map, discard_tile, -1, remove_zero=True)
        invisible_tiles_map = utils.map_increment(
            invisible_tiles_map, new_tile, -1, remove_zero=True)
        invisible_tiles_number -= 1
        hand_map = utils.map_increment(hand_map, new_tile, 1)
        round_n += 1

    ## borrowed from github this eval_func
    score = map_hand_eval_func(
        node._fixed_hand, hand_map, invisible_tiles_map,
        invisible_tiles_number,
        random.sample(invisible_tiles_map.keys(), k=1)[0])
    return score


def MCT_backpropagate(root, node, sim_result):
    # sim_result is the score
    # if not win until the last limit, sim_result = 0
    #print("Begin propogate")
    node_tmp = node
    while node_tmp != root:
        node_tmp.update_value(True, sim_result)
        p = node_tmp._parent
        if p != None:
            p.update_possible_action(node_tmp)
        node_tmp = p


def best_uct(node):
    if node._depth > node._max_depth:
        return None, None
    #print(node._depth)
    node.expand_children()
    if node._depth == 0:
        # is root,then only choose its children
        node.max_avg_score_of_child()
        for key, child in node._children.items():
            if child._visit_count == 0:
                return None, child
        max_s = float('-inf')
        max_c = None
        uct_scores = node.calculate_uct_child()
        for key, score in uct_scores.items():
            if score > max_s:
                max_s = score
                max_c = node._children[key]
        return None, max_c

    else:
        node.max_avg_score_of_tile_action()
        for tile, info in node._possible_actions.items():
            # first return unvisited child
            if info['num_visit'] == 0:
                if len(info['list_childs']) > 0:
                    return tile, random.sample(info['list_childs'], k=1)[0]
        node.calculate_uct_tile_action()
        max_score = float('-inf')
        max_tile_action = None
        for tile, info in node._possible_actions.items():
            if info['uct_score'] > max_score:
                max_score = info['uct_score']
                max_tile_action = tile
        if max_tile_action != None:
            return max_tile_action, random.sample(
                node._possible_actions[max_tile_action]['list_childs'], k=1)[0]
    return None, None

#=========================================================================================================
# move generator class of MCTS (implementation of MoveGenerator)
#=========================================================================================================

class MCTS_move(MoveGenerator):
    def __init__(self,
                 player_name,
                 c=2,
                 max_iteration=2000,
                 display_step=False):
        self._c = c
        self._max_iteration = max_iteration
        self.display_step = display_step
        super(MCTS_move, self).__init__(player_name)

    def decide_chow(self, player, new_tile, choices, neighbors, game):
        #print("BEGIN decide chow!!!!!")
        self.begin_decision()
        if self.display_step:
            self.print_game_board(player.fixed_hand, player.hand, neighbors,
                                  game)
            self.print_msg("Someone just discarded a %s. (%s)" % (
                new_tile.symbol, ", ".join([str(choice)
                                            for choice in choices])))

        best_choice = choices[0]
        choice_tag = 0
        hand_map, invisible_tiles_map, invisible_tiles_number = get_current_info(
            player, neighbors)
        rest_round = game.deck_size // 4
        # add the possible action into
        if rest_round > 0:
            root = node(None, None, None, None, rest_round)
            ## add not chow child:
            child_not_chow = node(
                player.fixed_hand,
                hand_map,
                invisible_tiles_map,
                invisible_tiles_number,
                rest_round,
                parent=root,
                c=self._c,
                depth=1)
            root.add_child(child_not_chow, -1)
            ## add chow choice child:
            ch = 0
            for choice in choices:
                new_fixed_hand = player.fixed_hand.copy()
                new_hand_map = hand_map.copy()
                new_invisible_map = invisible_tiles_map.copy()
                new_invisible_num = invisible_tiles_number
                tiles_for_melds = []

                for i in range(choice - 1, choice + 2):
                    tile = new_tile.generate_neighbor_tile(i)
                    tiles_for_melds.append(tile)
                    if tile != new_tile:
                        new_hand_map = utils.map_increment(
                            new_hand_map, tile, -1, remove_zero=True)
                    else:
                        utils.map_increment(
                            new_invisible_map, tile, -1, remove_zero=True)
                        new_invisible_num -= 1
                new_fixed_hand.append(("chow", False, tuple(tiles_for_melds)))
                new_child = node(
                    new_fixed_hand,
                    new_hand_map,
                    new_invisible_map,
                    new_invisible_num,
                    rest_round,
                    parent=root,
                    c=self._c,
                    depth=1)
                root.add_child(new_child, ch)
                ch += 1
            choice_tag = MCT_search(root, max_simulation=self._max_iteration)
        self.end_decision()
        if choice_tag == -1:
            self.print_msg("%s [%s] chooses not to Chow." % (self.player_name,
                                                             display_name))
            return False, None
        else:
            chow_tiles_str = ""
            best_choice = choices[choice_tag]
            for i in range(best_choice - 1, best_choice + 2):
                chow_tiles_str += new_tile.generate_neighbor_tile(i).symbol
                self.print_msg(
                    "%s [%s] chooses to Chow %s." %
                    (self.player_name, display_name, chow_tiles_str))
            return True, best_choice


    def decide_kong(self, player, new_tile, kong_tile, location, src,
                    neighbors, game):
        #print("BEGIN decide kong!!!!!")
        self.begin_decision()
        if self.display_step:
            self.print_game_board(player.fixed_hand, player.hand, neighbors,
                                  game)

        hand_map, invisible_tiles_map, invisible_tiles_number = get_current_info(
            player, neighbors)
        fixed_hand = player.fixed_hand
        rest_round = game.deck_size // 4

        # add the possible action into
        root = node(None, None, None, None, rest_round, c=self._c)

        # To kong
        kong_fixed_hand = list(fixed_hand)
        kong_map_hand = hand_map.copy()
        kong_map_remaining = invisible_tiles_map.copy()

        kong_tile_remaining = invisible_tiles_number - 1

        if location == "fixed_hand":
            utils.map_increment(
                kong_map_remaining, kong_tile, -1, remove_zero=True)
            for i in range(len(player.fixed_hand)):
                if kong_fixed_hand[i][0] == "pong" and kong_fixed_hand[i][2][
                        0] == kong_tile:
                    kong_fixed_hand[i] = ("kong", False,
                                          (kong_tile, kong_tile, kong_tile,
                                           kong_tile))
                    break
        else:
            is_secret = False
            if src == "steal":
                self.print_msg(
                    "Someone just discarded a %s." % kong_tile.symbol)
                utils.map_increment(
                    kong_map_hand, kong_tile, -3, remove_zero=True)
                utils.map_increment(
                    kong_map_remaining, kong_tile, -1, remove_zero=True)

            elif src == "draw":
                self.print_msg("You just drew a %s" % kong_tile.symbol)
                utils.map_increment(
                    kong_map_hand, kong_tile, -3, remove_zero=True)
                utils.map_increment(
                    kong_map_remaining, kong_tile, -1, remove_zero=True)

            elif src == "existing":
                self.print_msg("You have 4 %s in hand" % kong_tile.symbol)
                utils.map_increment(
                    kong_map_hand, kong_tile, -4, remove_zero=True)
                utils.map_increment(
                    kong_map_hand, new_tile, 1, remove_zero=True)
                utils.map_increment(
                    kong_map_remaining, new_tile, -1, remove_zero=True)

            kong_fixed_hand.append(("kong", is_secret, (kong_tile, kong_tile,
                                                        kong_tile, kong_tile)))

        result = False
        if game.deck_size // 4 > 0:
            node1 = node(
                kong_fixed_hand,
                kong_map_hand,
                kong_map_remaining,
                kong_tile_remaining,
                rest_round,
                depth=1,
                c=self._c,
                parent=root)
            root.add_child(node1, True)
            node2 = node(
                player.fixed_hand,
                hand_map,
                invisible_tiles_map,
                invisible_tiles_number,
                rest_round,
                depth=1,
                c=self._c,
                parent=root)
            root.add_child(node2, False)
            result = MCT_search(root, self._max_iteration)

        self.end_decision()
        if result:
            self.print_msg(
                "%s [%s] chooses to form a Kong %s%s%s%s." %
                (self.player_name, display_name, kong_tile.symbol,
                 kong_tile.symbol, kong_tile.symbol, kong_tile.symbol))
            return True
        else:
            self.print_msg(
                "%s [%s] chooses not to form a Kong %s%s%s%s." %
                (self.player_name, display_name, kong_tile.symbol,
                 kong_tile.symbol, kong_tile.symbol, kong_tile.symbol))
            return False

    def decide_pong(self, player, new_tile, neighbors, game):
        #print("BEGIN decide pong!!!!!")
        self.begin_decision()
        if self.display_step:
            self.print_game_board(player.fixed_hand, player.hand, neighbors,
                                  game)

        self.print_msg("Someone just discarded a %s." % new_tile.symbol)

        result = False
        rest_round = game.deck_size // 4
        if rest_round > 0:
            fixed_hand = player.fixed_hand
            hand_map, invisible_tiles_map, invisible_tiles_number = get_current_info(
                player, neighbors)

            invisible_tiles_map = utils.map_increment(
                invisible_tiles_map, new_tile, -1, remove_zero=True)
            invisible_tiles_number = invisible_tiles_number - 1

            root = node(None, None, None, None, rest_round, c=self._c)

            pong_fixed_hand = list(fixed_hand)
            pong_map_hand = hand_map.copy()
            pong_map_remaining = invisible_tiles_map.copy()

            utils.map_increment(pong_map_hand, new_tile, -2, remove_zero=True)
            pong_fixed_hand.append(("pong", False, (new_tile, new_tile,
                                                    new_tile)))

            node1 = node(
                pong_fixed_hand,
                pong_map_hand,
                pong_map_remaining,
                invisible_tiles_number,
                rest_round,
                parent=root,
                depth=1,
                c=self._c)
            root.add_child(node1, True)
            node2 = node(
                player.fixed_hand,
                hand_map,
                invisible_tiles_map,
                invisible_tiles_number,
                rest_round,
                parent=root,
                depth=1,
                c=self._c)
            root.add_child(node2, False)

            result = MCT_search(root, self._max_iteration)

        self.end_decision()
        if result:
            self.print_msg("%s [%s] chooses to form a Pong %s%s%s." %
                           (self.player_name, display_name, new_tile.symbol,
                            new_tile.symbol, new_tile.symbol))
            return True
        else:
            self.print_msg("%s [%s] chooses not to form a Pong %s%s%s." %
                           (self.player_name, display_name, new_tile.symbol,
                            new_tile.symbol, new_tile.symbol))
            return False

    def decide_drop_tile(self, player, new_tile, neighbors, game):
        #print("BEGIN decide drop tile!!!!!")
        hand_map, invisible_tiles_map, invisible_tiles_number = get_current_info(
            player, neighbors)

        self.begin_decision()
        if self.display_step:
            self.print_game_board(player.fixed_hand, player.hand, neighbors,
                                  game, new_tile)

        drop_tile = new_tile if new_tile is not None else player.hand[0]
        rest_round = game.deck_size // 4
        if rest_round > 0:
            fixed_hand = player.fixed_hand
            hand_map, invisible_tiles_map, invisible_tiles_number = get_current_info(
                player, neighbors)
            if new_tile is not None:
                hand_map = utils.map_increment(hand_map, new_tile, 1)
                invisible_tiles_map = utils.map_increment(
                    invisible_tiles_map, new_tile, -1, remove_zero=True)
                invisible_tiles_number -= 1

            root = node(
                fixed_hand,
                hand_map,
                invisible_tiles_map,
                invisible_tiles_number,
                rest_round,
                c=self._c,
                depth=0)
            root.expand_children()
            result = MCT_search(root, self._max_iteration)
            drop_tile = result

        self.print_msg("%s [%s] chooses to drop %s." %
                       (self.player_name, display_name, drop_tile.symbol))
        self.end_decision(True)
        return drop_tile

    def decide_win(self, player, grouped_hand, new_tile, src, score, neighbors,
                   game):
        #print("BEGIN decide win!!!!!")
        if self.display_step:
            if src == "steal":
                self.print_game_board(player.fixed_hand, player.hand,
                                      neighbors, game)
                self.print_msg(
                    "Someone just discarded a %s." % new_tile.symbol)
            else:
                self.print_game_board(
                    player.fixed_hand,
                    player.hand,
                    neighbors,
                    game,
                    new_tile=new_tile)

            self.print_msg("%s [%s] chooses to declare victory." %
                           (self.player_name, display_name))

            self.print_msg("You can form a victory hand of: ")
            utils.print_hand(player.fixed_hand, end=" ")
            utils.print_hand(grouped_hand, end=" ")
            self.print_msg("[%d]" % score)

        return True

    def reset_new_game(self):
        pass

    def print_msg(self, msg):
        if self.display_step:
            print(msg)


def get_current_info(player, neighbors):
    #print(type(neighbors))
    hand_map = {}
    invisible_tiles_map = Tile.get_tile_map(default_val=4)
    invisible_tiles_number = 36 * 3 + 7 * 4

    for tile in player.hand:
        hand_map = utils.map_increment(hand_map, tile, 1)
        invisible_tiles_map = utils.map_increment(invisible_tiles_map, tile,
                                                  -1)
        invisible_tiles_number -= 1

    players = list(neighbors) + [player]
    for p in players:
        for _, _, tiles in p.fixed_hand:
            for tile in tiles:
                invisible_tiles_map = utils.map_increment(
                    invisible_tiles_map, tile, -1)
                invisible_tiles_number -= 1

        for tile in p.get_discarded_tiles("unstolen"):
            invisible_tiles_map = utils.map_increment(invisible_tiles_map,
                                                      tile, -1)
            invisible_tiles_number -= 1

    return hand_map, invisible_tiles_map, invisible_tiles_number

###===============================================================================================================================
# Evaluation of hand tiles 
#### borrowed from github projet https://github.com/clarkwkw/mahjong-ai
##================================================================================================================================


display_name = "RNAIM"
suits = ["bamboo", "characters", "dots"]
s_chow, s_pong, s_future = 1, 1.2, 0.15


def one_faan_failing_criterion(chow_suits, pong_suits, is_honor, is_rgw):
    one_faan_criteria_1 = is_rgw

    # approximation to all_chows
    one_faan_criteria_2 = len(pong_suits) == 0 and not is_honor

    return not (one_faan_criteria_1 or one_faan_criteria_2)


def three_faan_failing_criterion(chow_suits, pong_suits, is_honor, is_rgw):
    failing_criteria = len(chow_suits) > 1 or (
        len(chow_suits) == 1 and
        (len(pong_suits) - (chow_suits[0] in pong_suits)) > 0)
    return failing_criteria


failing_criteria = {
    1: one_faan_failing_criterion,
    3: three_faan_failing_criterion
}


def map_hand_eval_func(fixed_hand,
                       map_hand,
                       map_remaining,
                       tile_remaining,
                       additional_tile=None):
    unique_tiles = []
    suit_tiles = {suit: [] for suit in suits}
    scoring_matrix = np.zeros((2, 3))
    max_score = 0
    base_score = len(fixed_hand)

    chow_suits = []
    pong_suits = []
    is_honor, is_rgw = False, False

    for meld_type, _, tiles in fixed_hand:
        if tiles[0].suit == "honor":
            is_honor = True
            if tiles[0].value in ["red", "green", "white"]:
                is_rgw = True
        elif meld_type == "chow":
            if tiles[0].suit not in chow_suits:
                chow_suits.append(tiles[0].suit)
        else:
            if tiles[0].suit not in pong_suits:
                pong_suits.append(tiles[0].suit)

    for tile, count in map_hand.items():
        if tile.suit == "honor":
            if count >= 2:
                matching_count = max(count, 3)
                map_hand[tile] -= matching_count
                base_score += s_pong * (matching_count / 3.0 +
                                        (1 - matching_count / 3.0) *
                                        (map_remaining.get(tile, 0) / 4.0))
                is_honor = True
                is_rgw = is_rgw or (tile.value in ["red", "green", "white"])
        else:
            suit_tiles[tile.suit].append(tile)

    if failing_criteria[HKRules.__score_lower_limit](chow_suits, pong_suits,
                                                     is_honor, is_rgw):
        return 0

    for i in range(len(suits)):
        suit = suits[i]
        for j in range(2):
            scoring_matrix[j, i] = eval_suit(map_hand, map_remaining,
                                             suit_tiles[suit], j > 0)

    if HKRules.__score_lower_limit == 1:
        return base_score + scoring_matrix[1, :].sum()

    else:
        # Possible cases reaching here:
        # 1. only 1 chow suit
        # 2. 0 chow suit with 0 pong suit
        # 3. 0 chow suit with 1 pong suit
        # 4. 0 chow suit with >1 pong suit
        if len(chow_suits) == 1:
            chow_suit_index = suits.index(chow_suits[0])
            return base_score + scoring_matrix[1, chow_suit_index]

        mixed_pong_score = 0
        for i in range(len(suits)):
            if scoring_matrix[0, i] > 0:
                mixed_pong_score += scoring_matrix[0, i]

        if len(pong_suits) == 0:
            return base_score + max(mixed_pong_score,
                                    scoring_matrix[1, :].max())

        elif len(pong_suits) == 1:
            pong_suit_index = suits.index(pong_suits[0])
            return base_score + max(mixed_pong_score,
                                    scoring_matrix[1, pong_suit_index])

        return base_score + mixed_pong_score


def eval_suit(map_hand,
              map_remaining,
              suit_tiles,
              is_chow,
              processing=0,
              tmp_score=0):
    max_score, matching_count, contribution = 0, 0, 0
    max_path = len(suit_tiles)

    for i in range(processing, len(suit_tiles)):
        tile = suit_tiles[i]

        if map_hand[tile] >= 2:
            matching_count = max(map_hand[tile], 3)
            map_hand[tile] -= matching_count
            contribution = s_pong * (matching_count / 3.0 +
                                     (1 - matching_count / 3.0) *
                                     (map_remaining.get(tile, 0) / 4.0))
            pong_score = eval_suit(
                map_hand,
                map_remaining,
                suit_tiles,
                is_chow,
                processing,
                tmp_score=tmp_score + contribution)
            if pong_score > max_score:
                max_score = pong_score
                max_path = i
            map_hand[tile] += matching_count

        if is_chow and tile.value <= 7:
            tile_triple = [
                tile,
                tile.generate_neighbor_tile(offset=1),
                tile.generate_neighbor_tile(offset=2)
            ]
            matching = [0, 0, 0]
            matching_count = 0
            chow_prob = 1
            for i in range(3):
                if map_hand.get(tile_triple[i], 0) > 0:
                    matching[i] = 1
                    matching_count += 1
                    map_hand[tile_triple[i]] -= 1
                else:
                    matching[i] = 0
                    chow_prob *= map_remaining.get(tile_triple[i], 0) / 4.0

            if matching_count >= 2 and chow_prob > 0:
                contribution = s_chow * matching_count / 3.0 * chow_prob
                chow_score = eval_suit(
                    map_hand,
                    map_remaining,
                    suit_tiles,
                    is_chow,
                    processing + 1,
                    tmp_score=tmp_score + contribution)
                if chow_score > max_score:
                    max_score = chow_score
                    max_path = i

            for i in range(3):
                if matching[i] > 0:
                    map_hand[tile_triple[i]] += 1

    if max_path == len(suit_tiles):
        for i in range(len(suit_tiles)):
            if map_hand[suit_tiles[i]] > 0:
                tmp_score += s_future * map_remaining.get(suit_tiles[i],
                                                          0) / 4.0

    return max(max_score, tmp_score)


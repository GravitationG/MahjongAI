import json
import random
#from TGLanguage import get_tile_name

import json

LANGUAGE_PACK_PATH = "resources/language.json"
LANGUAGE_DICT = None

with open(LANGUAGE_PACK_PATH, "r") as f:
	LANGUAGE_DICT = json.load(f)

def get_lang_codes():
	return list(LANGUAGE_DICT.keys())

def get_tile_name(lang_code, suit = None, value = None, is_short = True):
	if lang_code is None:
		lang_code = "EN"

	val = LANGUAGE_DICT[lang_code]["TILE_NAMES"][suit][value]

	if type(val) is list and len(val) == 2:
		return val[is_short]

	return val

def get_text(lang_code, text_code = ""):
	if lang_code is None:
		lang_code = "EN"

	return LANGUAGE_DICT[lang_code][text_code]


class Tile:
	def __init__(self, suit, value):
		self.__suit = suit
		self.__suit_id = suit_order.index(suit)
		try:
			self.__value = int(value)
		except ValueError:
			self.__value = value
		self.__symbol = tile_symbols[suit][str(value)]

	@property
	def suit(self):
		return self.__suit

	@property
	def symbol(self):
		return self.__symbol

	@property
	def value(self):
		return self.__value

	def get_display_name(self, lang_code, is_short = True):
		return get_tile_name(lang_code, self.__suit, str(self.__value), is_short)

	def __hash__(self):
		return hash("%s-%s"%(self.__suit, self.__value))

	def __str__(self):
		return "%s-%s"%(self.__suit, self.__value)

	def __eq__(self, other):
		if other is None:
			return False
		return (self.__suit == other.__suit) and (self.__value == other.__value)

	def __ne__(self, other):
		return not self == other

	def __lt__(self, other):
		if self.__suit_id < other.__suit_id:
			return True

		elif self.__suit == other.__suit:
			return self.__value < other.__value

		return False

	def generate_neighbor_tile(self, offset):
		if type(self.__value) is int and self.__value + offset >= 1 and self.__value + offset <= 9:
			tile = Tile(self.__suit, self.__value + offset)
			return tile
		return None

def get_tiles(shuffle = True):
	result_tiles = []
	for suit, collection in tile_symbols.items():
		for value, symbol in collection.items():
			for i in range(4):
				result_tiles.append(Tile(suit = suit, value = value))
	if shuffle:
		random.shuffle(result_tiles)
	return result_tiles

def get_tile_map(default_val = 4):
	result = {}
	for suit, collection in tile_symbols.items():
		for value, symbol in collection.items():
			tile = Tile(suit = suit, value = value)
			result[tile] = default_val

	return result

def get_tile_classification_map(default_val = None):
	result = {}
	for suit in tile_symbols:
		result[suit] = {}
		for value in tile_symbols[suit]:
			result[suit][value] = default_val
	return result

def get_suit_classification_map(default_val = None):
	result = {}
	for suit in tile_symbols:
		result[suit] = default_val
	return result

def convert_tile_index(val):
	if isinstance(val, Tile):
		return tile_index[val]

	return index_tile[val]  

with open("resources/tile_config.json", "r") as f:
	tile_config_dict = json.load(f)
	suit_order = tile_config_dict["suit_order"]
	tile_symbols = tile_config_dict["symbols"]
	tile_back_symbol = tile_config_dict["tile_back"]
	tile_map = {}
	tile_index = {}
	index_tile = []
	i = 0
	for suit in tile_symbols:
		tile_map[suit] = {}
		for value in tile_symbols[suit]:
			tile = Tile(suit, value)
			tile_map[suit][value] = tile

	for suit in suit_order:
		for value in tile_symbols[suit]:
			tile = tile_map[suit][value]
			tile_index[tile] = i
			index_tile.append(tile)
			i += 1


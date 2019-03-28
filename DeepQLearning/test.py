import json

LANGUAGE_PACK_PATH = ""
LANGUAGE_DICT = None

with open(LANGUAGE_PACK_PATH, "r") as f:
	LANGUAGE_DICT = json.load(f)
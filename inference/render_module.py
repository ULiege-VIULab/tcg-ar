import cv2
import os
import re
import sys
import json
import time
import imageio
import numpy as np
from multiprocessing import shared_memory

from core.config import *
from inference.registration_module import compute_destination_point

# Side (auxiliary) views render sprites smaller than the zenithal view: this factor is
# applied to the stored (zenithal-sized) sprite, e.g. 150% / 200% = 0.75.
SIDE_VIEW_ZOOM_RATIO = MODEL_ZOOM_SIDE_PERCENT / MODEL_ZOOM_DEFAULT_PERCENT

# NOTE: the Pokemon / sprite / card database builders that used to live here have
# moved to core.databases (single shared copy).  This module now only contains the
# runtime Game_state + Multi_frame_renderer + display_fps used during inference.

class Game_state:
    CARD_SLOT_INDEX = 0                 #index from the shareable list of the number of slot available for the card
    NUMBER_CARD_INDEX = 1               #index from the shareable list of the number of card stored

    POKEMON_CARD_ID_INDEX = 0           #index from the pokemon list of the ID of the pokemon card
    POKEMON_POKEDEX_NUMBER_INDEX_1 = 1  #index from the pokemon list of the national pokedex number of the first pokemon
    POKEMON_POKEDEX_NUMBER_INDEX_2 = 2  #index from the pokemon list of the national pokedex number of the second pokemon
    POKEMON_POKEDEX_NUMBER_INDEX_3 = 3  #index from the pokemon list of the national pokedex number of the third pokemon
    X_POS_INDEX = 4                     #index from the pokemon list of the x position of the pokemon
    Y_POS_INDEX = 5                     #index from the pokemon list of the y position of the pokemon
    SHINY_INDEX_1 = 6                   #index from the pokemon list if the first pokemon is shiny
    SHINY_INDEX_2 = 7                   #index from the pokemon list if the second pokemon is shiny
    SHINY_INDEX_3 = 8                   #index from the pokemon list if the third pokemon is shiny
    FEMALE_INDEX_1 = 9                  #index from the pokemon list if the first pokemon is female
    FEMALE_INDEX_2 = 10                 #index from the pokemon list if the second pokemon is female
    FEMALE_INDEX_3 = 11                 #index from the pokemon list if the third pokemon is female
    FORM_INDEX_1 = 12                   #index from the pokemon list of the form of the first pokemon
    FORM_INDEX_2 = 13                   #index from the pokemon list of the form of the second pokemon
    FORM_INDEX_3 = 14                   #index from the pokemon list of the form of the third pokemon
    MODEL_INDEX_1 = 15                  #index from the pokemon list of the model of the first pokemon
    MODEL_INDEX_2 = 16                  #index from the pokemon list of the model of the second pokemon
    MODEL_INDEX_3 = 17                  #index from the pokemon list of the model of the third pokemon
    NUMBER_POKEMON_INDEX = 18           #index from the pokemon list of the number of the pokemons in the card

    def __init__(self, existing_shm):
        self.existing_shm = existing_shm
        self.shared_list_variable = None

        try:
            database_file = open(POKEMON_DATABASE_FILE, "r", encoding = "utf-8")
        except FileNotFoundError:
            print("pokemon database file missing", file = sys.stderr)
            raise FileNotFoundError
        self.database = json.load(database_file)
        database_file.close()

        card_slot = 0
        number_card = 0
        self.pokemon_list = []

        if self.existing_shm:
            self.shared_list_variable = shared_memory.ShareableList(name = "shared_variable_pokemon")
            self.__add_card_slot(self.shared_list_variable[self.CARD_SLOT_INDEX])
        else:
            self.shared_list_variable = shared_memory.ShareableList([card_slot, number_card], name = "shared_variable_pokemon")
            self.__add_card_slot(6)
        
        self.form_dictionary = {
            "Base" : "",
            "Mega" : "mega",
            "Gigantamax" : "gigantamax",
            "Mega X" : "megax",
            "Mega Y" : "megay",
            "Alolan Form" : "alola",
            "Original Cap" : "originalcap",
            "Hoenn Cap" : "hoenncap",
            "Sinnoh Cap" : "sinnohcap",
            "Unova Cap" : "unovacap",
            "Kalos Cap" : "kaloscap",
            "Alola Cap" : "alolacap",
            "Partner Cap" : "partnercap",
            "World Cap" : "World Cap",  #this form does not exist for the animated database
            "Cosplay Pikachu" : "cosplay",
            "Pikachu Rock Star" : "rockstar",
            "Pikachu Belle" : "belle",
            "Pikachu Pop Star" : "popstar",
            "Pikachu, Ph.D." : "phd",
            "Pikachu Libre" : "libre",
            "Partner" : "Partner",  #this form does not exist for the animated database
            "Galarian Form" : "galar",
            "Hisuian Form" : "Hisuian Form",    #this form does not exist for the animated database
            "Paldean Form, Combat Breed" : "Paldean Form, Combat Breed",    #this form does not exist for the animated database
            "Paldean Form, Blaze Breed" : "Paldean Form, Blaze Breed",  #this form does not exist for the animated database
            "Paldean Form, Aqua Breed" : "Paldean Form, Aqua Breed",    #this form does not exist for the animated database
            "Spiky-eared" : "Spiky-eared",  #this form does not exist for the animated database
            "Spiky-eared" : "Spiky-eared",  #this form does not exist for the animated database
            "Paldean Form" : "Paldean Form",    #this form does not exist for the animated database
            "A" : "",   #this form exist but has no particular appellation
            "B" : "bravo",
            "C" : "charlie",
            "D" : "delta",
            "E" : "echo",
            "F" : "foxtrot",
            "G" : "golf",
            "H" : "hotel",
            "I" : "india",
            "J" : "juliet",
            "K" : "kilo",
            "L" : "lima",
            "M" : "mike",
            "N" : "november",
            "O" : "oscar",
            "P" : "papa",
            "Q" : "quebec",
            "R" : "romeo",
            "S" : "sierra",
            "T" : "tango",
            "U" : "uniform",
            "V" : "victor",
            "W" : "whiskey",
            "X" : "xray",
            "Y" : "yankee",
            "Z" : "zulu",
            "?" : "interrogation",
            "!" : "exclamation",
            "Sunny Form" : "sunny",
            "Rainy Form" : "rainy",
            "Snowy Form" : "snowy",
            "Normal Forme" : "",    #this form exist but has no particular appellation
            "Attack Forme" : "attack",
            "Defense Forme" : "defense",
            "Speed Forme" : "speed",
            "Plant Cloak" : "", #this form exist but has no particular appellation
            "Sandy Cloak" : "sandy",
            "Trash Cloak" : "trash",
            "Overcast Form" : "",    #this form exist but has no particular appellation
            "Sunshine Form" : "sunshine",
            "West Sea" : "",    #this form exist but has no particular appellation
            "East Sea" : "east",
            "Heat Rottom" : "heat",
            "Wash Rottom" : "wash",
            "Frost Rottom" : "frost",
            "Fan Rottom" : "fan",
            "Mow Rottom" : "mow",
            "Origin Forme" : "origin",
            "Altered Forme" : "",    #this form exist but has no particular appellation
            "Land Forme" : "",    #this form exist but has no particular appellation
            "Sky Forme" : "sky",
            "Fighting" : "fighting",
            "Flying" : "flying",
            "Poison" : "poison",
            "Ground" : "ground",
            "Rock" : "rock",
            "Bug" : "bug",
            "Ghost" : "ghost",
            "Steel" : "steel",
            "Fire" : "fire",
            "Water" : "water",
            "Grass" : "grass",
            "Electric" : "electric",
            "Psychic" : "psychic",
            "Ice" : "ice",
            "Dragon" : "dragon",
            "Dark" : "dark",
            "Fairy" : "fairy",
            "Red-Striped Form" : "",    #this form exist but has no particular appellation
            "Blue-Striped Form" : "blue",
            "White-Striped Form" : "White-Striped Form",    #this form does not exist for the animated database
            "Standard Mode" : "",    #this form exist but has no particular appellation
            "Zen Mode" : "zen",
            "Galarian Form, Standard Mode" : "galar",
            "Galarian Form, Zen Mode" : "galar-zen",
            "Spring Form" : "",    #this form exist but has no particular appellation
            "Summer Form" : "summer",
            "Autumn Form" : "autumn",
            "Winter Form" : "winter",
            "Incarnate Forme" : "",    #this form exist but has no particular appellation
            "Therian Forme" : "therian",
            "White Kyurem" : "white",
            "Black Kyurem" : "black",
            "Ordinary Form" : "",    #this form exist but has no particular appellation
            "Resolute Form" : "resolute",
            "Aria Forme" : "",    #this form exist but has no particular appellation
            "Pirouette Forme" : "pirouette",
            "Douse Drive" : "water",
            "Shock Drive" : "electric",
            "Burn Drive" : "fire",
            "Chill Drive" : "ice",
            "Ash-Greninja" : "active",
            "Icy Snow Pattern" : "",    #this form exist but has no particular appellation
            "Polar Pattern" : "polar",
            "Tundra Pattern" : "tundra",
            "Continental Pattern" : "continental",
            "Garden Pattern" : "garden",
            "Elegant Pattern" : "elegant",
            "Meadow Pattern" : "meadow",
            "Modern Pattern" : "modern",
            "Marine Pattern" : "marine",
            "Archipelago Pattern" : "archipelago",
            "High Plains Pattern" : "highplains",
            "Sandstorm Pattern" : "sandstorm",
            "River Pattern" : "river",
            "Monsoon Pattern" : "monsoon",
            "Savanna Pattern" : "savannah",
            "Sun Pattern" : "sun",
            "Ocean Pattern" : "ocean",
            "Jungle Pattern" : "jungle",
            "Fancy Pattern" : "fancy",
            "Poké Ball Pattern" : "pokeball",
            "Red Flower" : "",    #this form exist but has no particular appellation
            "Yellow Flower" : "yellow",
            "Orange Flower" : "orange",
            "Blue Flower" : "blue",
            "White Flower" : "white",
            "Natural Form" : "",    #this form exist but has no particular appellation
            "Heart Trim" : "heart",
            "Star Trim" : "star",
            "Diamond Trim" : "diamond",
            "Debutante Trim" : "debutante",
            "Matron Trim" : "matron",
            "Dandy Trim" : "dandy",
            "La Reine Trim" : "lareine",
            "Kabuki Trim" : "kabuki",
            "Pharaoh Trim" : "pharaoh",
            "Male" : "",    #this form exist but has no particular appellation
            "Female" : "",    #this form exist but has no particular appellation
            "Shield Forme" : "",    #this form exist but has no particular appellation
            "Blade Forme" : "blade",
            "Small Size" : "small",
            "Average Size" : "",    #this form exist but has no particular appellation
            "Large Size" : "large",
            "Super Size" : "super",
            "Neutral Mode" : "",    #this form exist but has no particular appellation
            "Active Mode" : "active",
            "50% Forme" : "",    #this form exist but has no particular appellation
            "10% Forme" : "10",
            "Complete Forme" : "complete",
            "Hoopa Confined" : "",    #this form exist but has no particular appellation
            "Hoopa Unbound" : "unbound",
            "Baile Style" : "",    #this form exist but has no particular appellation
            "Pom-Pom Style" : "pompom",
            "Pa'u Style" : "pau",
            "Sensu Style" : "sensu",
            "Midday Form" : "",    #this form exist but has no particular appellation
            "Midnight Form" : "midnight",
            "Dusk Form" : "dusk",
            "Solo Form" : "",    #this form exist but has no particular appellation
            "School Form" : "school",
            "Type: Normal" : "fighting",
            "Type: Fighting" : "fighting",
            "Type: Flying" : "flying",
            "Type: Poison" : "poison",
            "Type: Ground" : "ground",
            "Type: Rock" : "rock",
            "Type: Bug" : "bug",
            "Type: Ghost" : "ghost",
            "Type: Steel" : "steel",
            "Type: Fire" : "fire",
            "Type: Water" : "water",
            "Type: Grass" : "grass",
            "Type: Electric" : "electric",
            "Type: Psychic" : "psychic",
            "Type: Ice" : "ice",
            "Type: Dragon" : "dragon",
            "Type: Dark" : "dark",
            "Type: Fairy" : "fairy",
            "Meteor Form" : "",    #this form exist but has no particular appellation
            "Red Core" : "red",
            "Orange Core" : "orange",
            "Yellow Core" : "yellow",
            "Green Core" : "green",
            "Blue Core" : "blue",
            "Indigo Core" : "indigo",
            "Violet Core" : "violet",
            "Disguised Form" : "",    #this form exist but has no particular appellation
            "Busted Form" : "busted",
            "Dusk Mane" : "dusk-mane",
            "Dawn Wings" : "dawn-wings",
            "Ultra" : "ultra",
            "Original Color" : "original",
            "Gulping Form" : "gulping",
            "Gorging Form" : "gorging",
            "Amped Form" : "",    #this form exist but has no particular appellation
            "Low Key Form" : "low-key",
            "Phony Form" : "",    #this form exist but has no particular appellation
            "Antique Form" : "Antique Form",    #this form does not exist for the animated database
            "Vanilla Cream" : "",    #this form exist but has no particular appellation
            "Ruby Cream" : "ruby-cream",
            "Matcha Cream" : "matcha-cream",
            "Mint Cream" : "mint-cream",
            "Lemon Cream" : "lemon-cream",
            "Salted Cream" : "salted-cream",
            "Ruby Swirl" : "ruby-swirl",
            "Caramel Swirl" : "caramel-swirl",
            "Rainbow Swirl" : "rainbow-swirl",
            "Ice Face" : "",    #this form exist but has no particular appellation
            "Noice Face" : "noice",
            "Full Belly Mode" : "",    #this form exist but has no particular appellation
            "Hangry Mode" : "hangry-mode",
            "Hero of Many Battles" : "",    #this form exist but has no particular appellation
            "Crowned Sword" : "crowned-sword",
            "Crowned Shield" : "crowned-shield",
            "Eternamax" : "eternamax",
            "Single Strike Style" : "",    #this form exist but has no particular appellation
            "Gigantamax, Single Strike Style" : "gigantamax",
            "Rapid Strike Style" : "rapid-strike",
            "Gigantamax, Rapid Strike Style" : "rapid-strike-gigantamax",
            "Dada" : "dada"
        }

    def __del__(self):
        self.delete()

    def update_object(self):
        if len(self.pokemon_list) < self.shared_list_variable[self.CARD_SLOT_INDEX]:
            self.__add_card_slot(self.shared_list_variable[self.CARD_SLOT_INDEX] - len(self.pokemon_list))

    def update_state(self, new_state):
        nb_slot_available = self.shared_list_variable[self.CARD_SLOT_INDEX]
        #if we have to store more card than the number of slot available, we double the number of slot available
        while len(new_state) > nb_slot_available:
            self.__add_card_slot(nb_slot_available)
            nb_slot_available = self.shared_list_variable[self.CARD_SLOT_INDEX]

        for i, state in enumerate(new_state):
            self.pokemon_list[i][self.POKEMON_CARD_ID_INDEX] = state[self.POKEMON_CARD_ID_INDEX]
            nb_pokemon_in_card = len(state[1])
            for j in range(nb_pokemon_in_card):
                self.pokemon_list[i][self.POKEMON_POKEDEX_NUMBER_INDEX_1 + j] = state[1][j]
            self.pokemon_list[i][self.X_POS_INDEX] = state[2]
            self.pokemon_list[i][self.Y_POS_INDEX] = state[3]
            self.pokemon_list[i][self.SHINY_INDEX_1] = False
            self.pokemon_list[i][self.SHINY_INDEX_2] = False
            self.pokemon_list[i][self.SHINY_INDEX_3] = False
            self.pokemon_list[i][self.FEMALE_INDEX_1] = False
            self.pokemon_list[i][self.FEMALE_INDEX_2] = False
            self.pokemon_list[i][self.FEMALE_INDEX_3] = False
            self.pokemon_list[i][self.FORM_INDEX_1] = 1
            self.pokemon_list[i][self.FORM_INDEX_2] = 1
            self.pokemon_list[i][self.FORM_INDEX_3] = 1
            self.pokemon_list[i][self.MODEL_INDEX_1] = 1
            self.pokemon_list[i][self.MODEL_INDEX_2] = 1
            self.pokemon_list[i][self.MODEL_INDEX_3] = 1
            self.pokemon_list[i][self.NUMBER_POKEMON_INDEX] = nb_pokemon_in_card

        self.shared_list_variable[self.NUMBER_CARD_INDEX] = len(new_state)

    def update_pokemon_form(self, card_index, pokemon_index, shiny, female, form, model):
        self.update_object()

        self.pokemon_list[card_index][self.SHINY_INDEX_1 + pokemon_index] = shiny
        self.pokemon_list[card_index][self.FEMALE_INDEX_1 + pokemon_index] = female
        self.pokemon_list[card_index][self.FORM_INDEX_1 + pokemon_index] = form
        self.pokemon_list[card_index][self.MODEL_INDEX_1 + pokemon_index] = model

    def get_pokemon_path(self, card_index):
        self.update_object()
        paths = []

        for i in range(self.pokemon_list[card_index][self.NUMBER_POKEMON_INDEX]):
            path = ""
            type_of_file = self.pokemon_list[card_index][self.MODEL_INDEX_1 + i]

            #noPokemon
            if not(self.pokemon_list[card_index][self.POKEMON_POKEDEX_NUMBER_INDEX_1 + i]):
                return [NO_POKEMON_PATH]
            #2Danimated
            if type_of_file == 1:
                unwanted_pattern = ' |:|\''
                number = self.pokemon_list[card_index][self.POKEMON_POKEDEX_NUMBER_INDEX_1 + i] - 1
                name = self.database[number]["Forms"][0]["English_name"]
                name = re.sub(unwanted_pattern, "", name)
                name = re.sub("é", "e", name)
                name = re.sub("♀", "_f", name)
                name = re.sub("♂", "_m", name)

                path = POKEMON_2D_ANIMATED_MODEL_FOLDER + name

                form_number = self.pokemon_list[card_index][self.FORM_INDEX_1 + i] - 1
                form = self.__translate_form(self.database[number]["Forms"][form_number]["Form"])
                if form:
                    path = path + "-" + form

                if self.pokemon_list[card_index][self.FEMALE_INDEX_1 + i]:
                    path = path + "-f"
                if self.pokemon_list[card_index][self.SHINY_INDEX_1 + i]:
                    path = path + "-shiny"
                
                path = path + ".gif"

                if not os.path.exists(path):
                    path = re.sub("-f", "", path)

                if not os.path.exists(path):
                    type_of_file = 2
            #2D
            if type_of_file == 2:

                unwanted_pattern = ':'
                number = self.pokemon_list[card_index][self.POKEMON_POKEDEX_NUMBER_INDEX_1 + i] - 1

                form_number = self.pokemon_list[card_index][self.FORM_INDEX_1 + i] - 1

                name = self.database[number]["Forms"][form_number]["English_name"]
                form = self.database[number]["Forms"][form_number]["Form"]
                name = re.sub(unwanted_pattern, "", name)
                form = re.sub(unwanted_pattern, "", form)

                path = POKEMON_2D_MODEL_FOLDER + name + "_" + form + ".png"
            #3D
            if type_of_file == 3:
                pass

            paths.append(path)

        return paths
    
    def get_pokemon_paths(self):
        self.update_object()

        pokemon_paths = []
        for i in range(self.shared_list_variable[self.NUMBER_CARD_INDEX]):
            pokemon_paths = pokemon_paths + self.get_pokemon_path(i)

        return pokemon_paths

    def get_pokemon_possible_form(self, pokemon_pokedex_id):
        if pokemon_pokedex_id:
            pokemon_pokedex_id -= 1
            forms = self.database[pokemon_pokedex_id]['Forms']
            ret_val = []

            for form in forms:
                ret_val.append(form['Form'])

            return ret_val
        else:
            return pokemon_pokedex_id

    def get_card_location(self, card_index):
        self.update_object()

        return (self.pokemon_list[card_index][self.X_POS_INDEX], self.pokemon_list[card_index][self.Y_POS_INDEX])

    def get_pokemon_model(self, card_index, pokemon_index):
        self.update_object()

        return self.pokemon_list[card_index][self.MODEL_INDEX_1 + pokemon_index]

    def get_pokemon_card_id(self, card_index):
        self.update_object()

        return self.pokemon_list[card_index][self.POKEMON_CARD_ID_INDEX]

    def get_pokemon_name(self, card_index, pokemon_index):
        self.update_object()
        if self.pokemon_list[card_index][self.POKEMON_POKEDEX_NUMBER_INDEX_1 + pokemon_index]:
            number = self.pokemon_list[card_index][self.POKEMON_POKEDEX_NUMBER_INDEX_1 + pokemon_index] - 1

            return self.database[number]["Forms"][0]["English_name"]
        else:
            return None

    def get_item_id(self, card_index):
        return (self.pokemon_list[card_index][self.POKEMON_CARD_ID_INDEX], self.pokemon_list[card_index][self.X_POS_INDEX], self.pokemon_list[card_index][self.Y_POS_INDEX])

    def get_number_of_card(self):
        self.update_object()

        return self.shared_list_variable[self.NUMBER_CARD_INDEX]

    def get_number_of_pokemon(self, card_index):
        self.update_object()

        return self.pokemon_list[card_index][self.NUMBER_POKEMON_INDEX]

    def is_pokemon_card(self, card_index):
        """True iff the card carries at least one real Pokemon (a non-zero national
        Pokedex number in any slot). Trainer/Energy and card-back entries keep a
        Pokemon slot with a zero Pokedex number -- the same "noPokemon" signal used
        by get_pokemon_path -- so they return False here."""
        self.update_object()

        for j in range(self.pokemon_list[card_index][self.NUMBER_POKEMON_INDEX]):
            if self.pokemon_list[card_index][self.POKEMON_POKEDEX_NUMBER_INDEX_1 + j]:
                return True
        return False

    #Give the string translation between the form provide by the database and the form use as name for the 2D animated image file.
    def __translate_form(self, form):
        if form not in self.form_dictionary:
            return form
        else:
            return self.form_dictionary[form]

    def __create_variable_list_name_from_ID(self, id):
        return "shared_pokemon_" + str(id)

    def __add_card_slot(self, number_slot_to_create):
        for _ in range(number_slot_to_create):
            if self.existing_shm:
                number = len(self.pokemon_list)
                shared_list_variable = shared_memory.ShareableList(name = self.__create_variable_list_name_from_ID(number))
                self.pokemon_list.append(shared_list_variable)
            else:
                shared_list_variable = shared_memory.ShareableList(["void0000000000000000", 0, 0, 0, 0.0, 0.0, False, False, False, False, False, False, 1, 1, 1, 1, 1, 1, 0], 
                                                                    name = self.__create_variable_list_name_from_ID(self.shared_list_variable[self.CARD_SLOT_INDEX]))
                self.pokemon_list.append(shared_list_variable)
                self.shared_list_variable[self.CARD_SLOT_INDEX] += 1

    def delete(self):
        if self.existing_shm:
            for i in range(len(self.pokemon_list)):
                self.pokemon_list[i].shm.close()
            self.shared_list_variable.shm.close()
        else:
            if self.shared_list_variable:
                for i in range(self.shared_list_variable[self.CARD_SLOT_INDEX]):
                    self.pokemon_list[i].shm.close()
                    self.pokemon_list[i].shm.unlink()
                self.shared_list_variable.shm.close()
                self.shared_list_variable.shm.unlink()

class Multi_frame_renderer:
    def __init__(self, number_of_view):
        self.number_of_view = number_of_view
        self.pokemon_dict = dict()
        self.pokemon_original_models = []
        self.pokemon_models = []
        self.pokemon_models_side = []   # pre-scaled for side views (SIDE_VIEW_ZOOM_RATIO)
        self.gif_duration = []
        self.num_frames_in_gif = []
        self.current_frame_num = []
        self.time_elapsed = []
        self.time = np.full((self.number_of_view,), time.time())
        # Sprite path cache: avoids regex + filesystem calls every frame.
        # Key = tuple of the shared-memory fields that determine the path.
        self._path_cache: dict = {}
        self._path_cache_key: dict = {}
    
    def load_models(self, files_to_load):
        #remove entry from pokemon_dict that is not in files_to_load
        #remove models from pokemon_models that has been removed from the dict
        #remove entry from files_to_load that is in pokemon_dict
        dict_index = 0
        for key in list(self.pokemon_dict.keys()):
            try:
                list_index = files_to_load.index(key)
                files_to_load.pop(list_index)
                self.pokemon_dict[key] = dict_index
                dict_index += 1
            except ValueError:
                self.pokemon_dict.pop(key)
                self.pokemon_original_models.pop(dict_index)
                self.pokemon_models.pop(dict_index)
                self.pokemon_models_side.pop(dict_index)
                self.gif_duration.pop(dict_index)
                self.num_frames_in_gif.pop(dict_index)
                self.current_frame_num.pop(dict_index)
                self.time_elapsed.pop(dict_index)

        #add entry to pokemon_dict that is in files_to_load
        #add models to pokemon_models and meta data
        for file in files_to_load:
            if type(self.pokemon_dict.get(file)) != int:
                index =  len(self.pokemon_dict)
                self.pokemon_dict[file]= index

                filename, file_extension = os.path.splitext(file)
                if file_extension == ".gif":
                    gif = imageio.mimread(file, '.gif', **{"mode":"RGBA"})
                    num_frames = len(gif)
                    meta = imageio.v3.immeta(file)
                    duration = meta.get("duration")
                    #transform into sec
                    duration /= 1000

                    img = np.array([cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA) for img in gif])
                else:
                    gif = imageio.mimread(file, None, **{"mode":"RGBA"})
                    num_frames = 1
                    duration = None

                    img = np.array([cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA) for img in gif])

                self.pokemon_original_models.append(img)
                self.pokemon_models.append(img)
                self.pokemon_models_side.append(img)   # placeholder, filled below or by zoom_models
                self.gif_duration.append(duration)
                self.num_frames_in_gif.append(num_frames)
                self.current_frame_num.append(np.zeros((self.number_of_view), dtype = np.int32))
                self.time_elapsed.append(0)

                # 2D animated models default to MODEL_ZOOM_DEFAULT_PERCENT so they are
                # visible at the wanted size without the user touching the Zoom slider
                # (the slider starts at the same value and rescales from the original).
                if file_extension == ".gif" and MODEL_ZOOM_DEFAULT_PERCENT != 100:
                    self.zoom_models([file], MODEL_ZOOM_DEFAULT_PERCENT / 100)
                else:
                    self._build_side_model(len(self.pokemon_dict) - 1)

    def _build_side_model(self, pokemon_index):
        """Pre-scale pokemon_models[pokemon_index] by SIDE_VIEW_ZOOM_RATIO for side views."""
        if SIDE_VIEW_ZOOM_RATIO == 1.0:
            self.pokemon_models_side[pokemon_index] = self.pokemon_models[pokemon_index]
            return
        src = self.pokemon_models[pokemon_index]
        h, w = src.shape[1], src.shape[2]
        sh = max(1, int(round(h * SIDE_VIEW_ZOOM_RATIO)))
        sw = max(1, int(round(w * SIDE_VIEW_ZOOM_RATIO)))
        self.pokemon_models_side[pokemon_index] = np.stack(
            [cv2.resize(f, (sw, sh), interpolation=cv2.INTER_LINEAR) for f in src]
        )

    def zoom_models(self, pokemon_paths, zoom):
        pokemon_index = None

        for pokemon_path in pokemon_paths:
            if type(self.pokemon_dict.get(pokemon_path)) == int:
                pokemon_index = self.pokemon_dict.get(pokemon_path)

            if pokemon_index == None:
                    continue

            zoom_tuple = (zoom, zoom)
            shape = self.pokemon_original_models[pokemon_index][0].shape[0:2]
            shape = np.flip(np.multiply(shape, zoom_tuple), 0).astype(int)

            new_store_shape = self.pokemon_original_models[pokemon_index].shape
            new_store_shape = np.multiply(new_store_shape, (1, zoom, zoom, 1)).astype(int)

            pokemon_model = self.pokemon_original_models[pokemon_index]
            self.pokemon_models[pokemon_index] = np.zeros(new_store_shape)

            for i in range(self.num_frames_in_gif[pokemon_index]):
                zoomed_model = cv2.resize(pokemon_model[i], shape, interpolation = cv2.INTER_LINEAR)
                self.pokemon_models[pokemon_index][i] = zoomed_model

            self._build_side_model(pokemon_index)

    def _cached_pokemon_path(self, game_state, i):
        """Return sprite paths for card i, recomputing only when the card's fields change."""
        pl = game_state.pokemon_list[i]
        key = (
            pl[Game_state.NUMBER_POKEMON_INDEX],
            pl[Game_state.POKEMON_POKEDEX_NUMBER_INDEX_1],
            pl[Game_state.POKEMON_POKEDEX_NUMBER_INDEX_2],
            pl[Game_state.POKEMON_POKEDEX_NUMBER_INDEX_3],
            pl[Game_state.MODEL_INDEX_1],
            pl[Game_state.MODEL_INDEX_2],
            pl[Game_state.MODEL_INDEX_3],
            pl[Game_state.FORM_INDEX_1],
            pl[Game_state.FORM_INDEX_2],
            pl[Game_state.FORM_INDEX_3],
            pl[Game_state.FEMALE_INDEX_1],
            pl[Game_state.FEMALE_INDEX_2],
            pl[Game_state.FEMALE_INDEX_3],
            pl[Game_state.SHINY_INDEX_1],
            pl[Game_state.SHINY_INDEX_2],
            pl[Game_state.SHINY_INDEX_3],
        )
        if self._path_cache_key.get(i) != key:
            self._path_cache[i] = game_state.get_pokemon_path(i)
            self._path_cache_key[i] = key
        return self._path_cache[i]

    def render_frame(self, id_frame, frame, game_state, homography_matrix, bottom_anchor=False):
        # bottom_anchor: place the BOTTOM of each sprite on the card (used for the side
        # views, so the model stands on the card) instead of the sprite's middle (the
        # zenithal view keeps the centred placement, bottom_anchor=False).
        game_state.update_object()
        number_card = game_state.get_number_of_card()

        old_time = self.time[id_frame]
        self.time[id_frame] = time.time()

        for i in range(len(self.pokemon_models)):
            if self.num_frames_in_gif[i] != 1:
                self.time_elapsed[i] = self.time_elapsed[i] + self.time[id_frame] - old_time
                if self.time_elapsed[i] > self.gif_duration[i]:
                    self.current_frame_num[i][id_frame] = (self.current_frame_num[i][id_frame] + int(self.time_elapsed[i] / self.gif_duration[i])) % self.num_frames_in_gif[i]
                    self.time_elapsed[i] = self.time_elapsed[i] % self.gif_duration[i]

        # Reload any sprite models whose path changed (e.g. shiny / form toggle).
        _all_sprite_paths = []
        _needs_reload = False
        for _ii in range(number_card):
            for _f in self._cached_pokemon_path(game_state, _ii):
                if _f != NO_POKEMON_PATH:
                    _all_sprite_paths.append(_f)
                    if not isinstance(self.pokemon_dict.get(_f), int):
                        _needs_reload = True
        if _needs_reload:
            self.load_models(list(_all_sprite_paths))

        # Painter's order: a card higher up in the image is further from the camera, so
        # draw far (top of image) -> near (bottom) and nearer sprites overlay the ones
        # behind them. The order is computed per view from each card's projected
        # position, so it adapts to every camera, not just the zenithal one.
        draw_order = []
        for i in range(number_card):
            board_x, board_y = game_state.get_card_location(i)
            projected = compute_destination_point([board_x, board_y, 1], homography_matrix)
            draw_order.append((projected[1], i, projected, board_x))
        draw_order.sort(key = lambda item: item[0])

        for _projected_row, i, projected, board_x in draw_order:
            files = self._cached_pokemon_path(game_state, i)
            pokemon_index = None

            # Sprites on the left half of the board are mirrored horizontally so the two
            # players' Pokemon face each other. Based on the board position, so the flip
            # is identical in every view.
            flip_sprite = board_x < WIDTH / 2

            for j in range(len(files)):
                # Non-Pokemon cards (Trainer/Energy) have no model: do NOT draw the
                # pokeball placeholder on the AR frame. The card still appears in the
                # GUI side panel (Card_menu lists every detected card).
                if files[j] == NO_POKEMON_PATH:
                    continue

                if type(self.pokemon_dict.get(files[j])) == int:
                    pokemon_index = self.pokemon_dict.get(files[j])

                if pokemon_index == None:
                    continue

                offsets = projected[0:2]

                # Resolve the sprite frame first, because the placement below depends on
                # its final size. Side views render the sprite smaller than the zenithal
                # view (MODEL_ZOOM_SIDE_PERCENT vs MODEL_ZOOM_DEFAULT_PERCENT).
                current_frame_number = self.current_frame_num[pokemon_index][id_frame]
                if bottom_anchor and SIDE_VIEW_ZOOM_RATIO != 1.0:
                    model_frame = self.pokemon_models_side[pokemon_index][current_frame_number]
                else:
                    model_frame = self.pokemon_models[pokemon_index][current_frame_number]
                #mirror left-side sprites so the two players' Pokemon face each other
                if flip_sprite:
                    model_frame = model_frame[:, ::-1, :]
                model_h, model_w = model_frame.shape[0], model_frame.shape[1]

                #by removing half the pokemon model shape from the offset the added pokemon
                #is centred on the wanted offset (else the top-left corner would be there).
                if len(files) == 1:
                    x_offset = max(0, int(offsets[1] - model_h / 2))
                    y_offset = max(0, int(offsets[0] - model_w / 2))

                elif len(files) == 2:
                    if j == 0:                       # left of the card centre
                        x_offset = max(0, int(offsets[1] - model_h / 2))
                        y_offset = max(0, int(offsets[0] - model_w))
                    elif j == 1:                     # right of the card centre
                        x_offset = max(0, int(offsets[1] - model_h / 2))
                        y_offset = max(0, int(offsets[0]))

                elif len(files) == 3:
                    if j == 0:                       # top-left
                        x_offset = max(0, int(offsets[1] - model_h))
                        y_offset = max(0, int(offsets[0] - model_w))
                    elif j == 1:                     # top-right
                        x_offset = max(0, int(offsets[1] - model_h))
                        y_offset = max(0, int(offsets[0]))
                    elif j == 2:                     # bottom-centre
                        x_offset = max(0, int(offsets[1]))
                        y_offset = max(0, int(offsets[0] - model_w / 2))

                # Side views: shift the sprite up by half its height so its bottom edge
                # (rather than its middle) sits on the card -- the model then stands on
                # the card. Horizontal centring is unchanged.
                if bottom_anchor:
                    x_offset = max(0, int(x_offset - model_h / 2))

                height = x_offset + model_h
                width = y_offset + model_w

                if height > HEIGHT:
                    offset = height - HEIGHT
                    x_offset -= offset
                    height -= offset

                if width > WIDTH:
                    offset = width - WIDTH
                    y_offset -= offset
                    width -= offset

                mask = model_frame[:, :, CHANNELS].astype(bool)
                roi = frame[x_offset:height, y_offset:width]
                roi[mask] = model_frame[:, :, :CHANNELS][mask]
        
        return frame

"""
Change the color space of an image to HSV.
Input:  -frame: a matrix representing an image.
Output: a matrix representing an image.
"""
def toHSV(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def display_fps(frame, time_elapsed_array):
    font = cv2.FONT_HERSHEY_SIMPLEX

    time_elapsed = np.mean(time_elapsed_array)

    fps = int(1/time_elapsed)

    return cv2.putText(frame, str(fps), (1875, 25), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

"""
Test the combination of the classes Game_state and Multi_frame_renderer to render AR frame.
Input:  None
Output: None
"""
def test():
    my_game_state = Game_state(False)
    
    state = [["ipcp0001", [6], 200, 200],         #dracaufeu
            ["ipcp0002", [201], 400, 400],        #zarbi
            ["ipcp0003", [1025], 800, 800],       #pechaminus
            ["ipcp0004", [487], 600, 608],        #giratina
            ["ipcp0005", [493], 0, 800],          #arceus
            ["ipcp0006", [800], 0, 800],          #necrozma
            ["ipcp0007", [1], 0, 0],              #bulbisar
            ["ipcp0008", [6], 500, 1200]          #dracaufeu
            ]

    my_game_state.update_state(state)
    my_game_state.update_pokemon_form(0,0,True,False,3,1)

    myrenderer = Multi_frame_renderer(1)
    
    pokemon = my_game_state.get_pokemon_paths()
    myrenderer.load_models(pokemon)

    myrenderer.zoom_models(my_game_state.get_pokemon_path(6), 1.5)

    #webcam_feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    webcam_feed = cv2.VideoCapture(0, cv2.CAP_MSMF)
    webcam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    webcam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    webcam_feed.set(cv2.CAP_PROP_FPS, FRAMERATE)

    time_length = 60
    time_elapsed_array = np.zeros((time_length,))
    new_time = time.time()
    nb_frame_produce = 0

    while True:
        ret, frame = webcam_feed.read()
        if frame.shape != (HEIGHT, WIDTH, CHANNELS):
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)
        
        old_time = new_time
        final_img = myrenderer.render_frame(0, frame, my_game_state, np.eye(3,3))
        new_time = time.time()
        nb_frame_produce = (nb_frame_produce + 1) % time_length
        time_elapsed_array[nb_frame_produce] = new_time - old_time
        final_img = display_fps(final_img, time_elapsed_array)

        cv2.imshow("cam", final_img)
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord('q'):
            state = [["ipcp0001", [9], 200, 200], #tortank
            ["ipcp0002", [208], 400, 400],        #stelix
            ["ipcp0005", [569], 0, 800],          #miasmax
            ["ipcp0001", [9], 500, 1200],          #tortank
            ["ipcp0009", [9], 900, 1200]          #tortank
            ]
            my_game_state.update_state(state)
            my_game_state.update_pokemon_form(4,0,True,False,3,2)

            pokemon = my_game_state.get_pokemon_paths()

            myrenderer.load_models(pokemon)
            myrenderer.zoom_models(my_game_state.get_pokemon_path(3),0.5)

    cv2.destroyAllWindows()
    webcam_feed.release()

#importing 2d database from https://projectpokemon.org/home/docs/spriteindex_148/3d-models-generation-1-pok%C3%A9mon-r90/
if __name__ == '__main__':
    #create a json file
    #create_database()

    #download pokemon gif
    #create_2D_animated_database()

    #download pokemon png
    #create_2D_database()

    #download pokemon cards
    #create_card_file()
    
    #use Gamestate and multi_frame_renderer classes to add multiple pokemon to a camera feed
    test()
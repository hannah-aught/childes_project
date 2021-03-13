import nltk
import glob
import os
import json
import re
from collections import defaultdict
nltk.download('punkt')

BILINGUAL_PATHS = glob.glob('./data/Bilingual/**/*cha')
MONOLINGUAL_PATHS = glob.glob('./data/Monolingual/**/*.cha')
OUTPUT_DIR = './output/'
BILINGUAL_DIR = OUTPUT_DIR + 'bilingual/'
MONOLINGUAL_DIR = OUTPUT_DIR + 'monolingual/'
BILINGUAL_OUT = BILINGUAL_DIR + 'bilingual_data.json'
MONOLINGUAL_OUT = MONOLINGUAL_DIR + 'monolingual_data.json'

# function to get the participants from a transcript file
# @text: the text from the file, NOT split by lines
# @return: an array of tuples, each containing the name of the child,
# and the corresponding line label used in the transcript
def get_target_participants(text):
    participants = []
    pattern = re.compile('^@Participants * (CHI[0-9]?)([a-zA-Z]+) Target_Child')
    results = re.findall(pattern, text)

    for result in results:
        participants.append((result.groups(1), result.groups(2)))

    return participants

# get the development stage for the transcript using the file name
# @file_name: the file name (which is the child's age)
# @return: the development stage
def get_stage(file_name):
    # The file name can only begin with 01, 02, or 03 in our data
    # if it's 1, they're a stage 1 child
    if file_name[:2] == '01':
        return 1
    # if it's 2 they're stage 3
    elif file_name[:2] == '02':
        return 3
    # otherwise they're stage 5
    else:
        return 5

# Get the tags 
# @text: the file contents, split into lines
# @return: the tags for each child's speech
def get_tags(text):
    pass

# Tokenize the child's speech
# @text: the file contents, split into lines
# @return: the tokens for each child's speech
def get_tokens(text):
    pass

# Get the types used by the child in a transcript
# @text: the file contents, split into lines
# @return: the types for each transcript
# TODO: should this be called over each stage rather than over each 
# transcript individually? Doing it over each transcript would mean
# that we're collecting the average number of types used per interaction
def get_types(text):
    pass

# Get the average length of utterance for a transcript
# @text: a transcript, split into lines
# returns the mean utterance length for the child's speech in the transcript
def get_mean_utterance_length(text):
    pass

# Wrapper function to get the stats for each monolingual and bilingual transcript
# @paths: the glob paths for each 
# @return: a dict containing the tokens, types, tags, participants, 
# development stages, number of tokens, number of types, number of (distinct) 
# tags, and mean utterance length for each transcript
def get_stats(paths):
    stats_obj = defaultdict(lambda: dict())

    for path in paths:
        with open(path, 'r') as f:
            data = f.read()
            participants = get_target_participants(data)
            stage = get_stage(data)
            tokens = get_tokens(data, participants)
            types = get_types(data, participants)
            tags = get_tags(data, participants)
            mean_utterance_length = get_mean_utterance_length(data, participants)


        for participant in participants:
            stats_obj[path].append({
                'participant': participant,
                'stage': stage,
                'tokens': tokens,
                'types': types,
                'tags': tags,
                'num_tokens': len(tokens),
                'num_types': len(types),
                'num_tags': len(tags),
                'mean_utterance_length': mean_utterance_length
            })

def generate_plots(data):
    pass

def main():
    # make the output file if it doesn't exist
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    monolingual_stats = get_stats(MONOLINGUAL_PATHS)
    bilingual_stats = get_stats(BILINGUAL_PATHS)

    generate_plots(monolingual_stats, MONOLINGUAL_DIR)
    generate_plots(bilingual_stats, BILINGUAL_DIR)

    json.dump(monolingual_stats, MONOLINGUAL_OUT)
    json.dump(bilingual_stats, BILINGUAL_OUT)

if __name__ == "__main__":
    main()
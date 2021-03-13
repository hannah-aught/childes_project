#%%
import nltk
import glob
import os
import json
import re
from collections import defaultdict
import seaborn as sns
import pandas as pd
nltk.download('punkt')

#%%
PATHS = glob.glob('./data/*/*/*cha')
OUTPUT_DIR = './output/'
BILINGUAL_DIR = OUTPUT_DIR + 'bilingual/'
MONOLINGUAL_DIR = OUTPUT_DIR + 'monolingual/'
BILINGUAL_OUT = BILINGUAL_DIR + 'bilingual_data.json'
MONOLINGUAL_OUT = MONOLINGUAL_DIR + 'monolingual_data.json'
PLOTS_DIR = OUTPUT_DIR + 'plots/'

#%%
# function to get the participants from a transcript file
# @text: the text from the file, NOT split by lines
# @return: an array of tuples, each containing the name of the child,
# and the corresponding line label used in the transcript
def get_target_participants(text):
    participants = []
    pattern = re.compile(r'(CHI[0-9]?) ([a-zA-Z]+) Target_Child')
    matches = pattern.finditer(text)

    for match in matches:
        participants.append((match.group(1), match.group(2)))

    return participants

#%%
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
    # otherwise it's (implicitly) 3 and they're stage 5
    else:
        return 5

#%%
# Get the tags 
# @text: the file contents, split into lines
# @return: the tags for each child's speech
def get_tags(text):
    pass

#%%
# Tokenize the child's speech
# @text: the file contents, split into lines
# @return: the tokens for each child's speech
def get_tokens(text):
    pass

#%%
# Get the types used by the child in a transcript
# @text: the file contents, split into lines
# @return: the types for each transcript
# TODO: should this be called over each stage rather than over each 
# transcript individually? Doing it over each transcript would mean
# that we're collecting the average number of types used per interaction
def get_types(text):
    pass

#%%
# Get the average length of utterance for a transcript
# @text: a transcript, split into lines
# returns the mean utterance length for the child's speech in the transcript
def get_mean_utterance_length(text):
    pass

#%%
# Wrapper function to get the stats for each monolingual and bilingual transcript
# @paths: the glob paths for each 
# @tag: a tag indicating if this is monolingual or bilingual
# @return: a dict containing the tokens, types, tags, participants, 
# development stages, number of tokens, number of types, number of (distinct) 
# tags, and mean utterance length for each transcript
def get_stats():
    stats = []

    for path in PATHS:
        with open(path, 'r') as f:
            child_class = path.split('/')[2]
            data = f.read()
            participants = get_target_participants(data)
            stage = get_stage(data)
            tokens = get_tokens(data)
            types = get_types(data)
            tags = get_tags(data)
            mean_utterance_length = get_mean_utterance_length(data)

        for participant in participants:
            stats.append((child_class,
                          participant,
                          stage,
                          tokens,
                          types,
                          tags,
                          len(tokens),
                          len(types),
                          len(tags),
                          mean_utterance_length))

        df = pd.DataFrame(stats, columns=['child_class', 'name', 'stage', 'tokens', 'types', 'tags', 'num_tokens', 'num_types', 'num_tags', 'mean_utterance_length'])
        return df

#%%
# function to generate the development curves from the collected data
# @data: an array of the stat dicts for monolingual and bilingual data
# @path: the path to write the plots to
def generate_plots(monolingual_data, bilingual_data, path):
    mono_types_hist = sns.histplot(monolingual_data, x='stage', y='num_types')
    bi_types_hist = sns.histplot(bilingual_data, x='stage', y='num_types')

    mono_tokens_hist = sns.histplot(monolingual_data, x='stage', y='num_tokens')
    bi_tokens_hist = sns.histplot(bilingual_data, x='stage', y='num_tokens')

    mono_utterance_length = sns.histplot(monolingual_data, x='stage', y='mean_utterance_length')
    bi_utterance_length = sns.histplot(bilingual_data, x='stage', y='mean_utterance_length')
    pass

#%%
# Wrapper function to write a stat dict to a file
# @data: the stat dict
# @path: the path to the file to write
def write_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

#%%
# Function to make the output directories for monolingual and bilingual data
# if they don't already exist
def make_output_dirs():
    # check if each dir exists, then make it if not
    if not os.path.exists(MONOLINGUAL_DIR):
        os.makedirs(MONOLINGUAL_DIR)
    if not os.path.exists(BILINGUAL_DIR):
        os.makedirs(BILINGUAL_DIR)


#%%
make_output_dirs()

#%%
df = get_stats()

#%%
def main():
    make_output_dirs()

    monolingual_df = get_stats(MONOLINGUAL_PATHS, tag='monolingual')
    bilingual_df = get_stats(BILINGUAL_PATHS, tag='bilingual')

    generate_plots(monolingual_stats, bilingual_stats, PLOTS_DIR)

    write_data(monolingual_stats, MONOLINGUAL_OUT)
    write_data(bilingual_stats, BILINGUAL_OUT)

if __name__ == "__main__":
    main()
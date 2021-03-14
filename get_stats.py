#%%
import nltk
import glob
import os
import json
import re
from collections import defaultdict, Counter
import seaborn as sns
import pandas as pd
nltk.download('punkt')

#%%
PATHS = glob.glob('./data/*/*/*.cha')
PATHS.sort()
OUTPUT_DIR = './output/'
OUTPUT_PATH = OUTPUT_DIR + 'language_data.csv'
PLOTS_DIR = OUTPUT_DIR + 'plots/'

#%%
# function to get the participants from a transcript file
# @text: the text from the file, NOT split by lines
# @return: a list of tuples, each containing the name of the child,
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
def get_stage(file_path):
    file_name = file_path.split('/')[-1]
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
# Function to get the lines with POS tags for an utterance
# @lines: the transcript split by newline
# @i: the first line the child is speaking on for this utterance
# @return: a list of the lines containing tags for the utterance
def get_tag_lines(lines, i):
    tag_lines = []
    
    for j, line in enumerate(lines[i+1:]):
        if len(line) >= 1 and line[0] == '*':
            break

        if '%mor' in line:
            tag_lines.append(line)

            for tag_line in lines[i+j+2:]:
                if tag_line[0] == '%' or tag_line == '*':
                    break

                tag_lines.append(tag_line)
    
    return tag_lines

#%%
# Get the tags 
# @text: the file contents, split into lines
# @return: the tags for each child's utterance
def get_tags(lines):
    tags = []
    pattern = re.compile(r'(%mor:\t|\|[^\s~]+\s*|:\w+|~|\+\.+)')

    for line in lines:
        tags += [x for x in re.sub(pattern, ' ', line).split() if x.isalpha()]

    return tags

#%%
# Function to get the errors noted in for an utterance
# @lines: the transcript split by newline
# @i: the first index of the child talking
# @return: a list of tuples containing the errors and fixes in the form (error, fix)
def get_errors(lines, i):

    for line in lines[i+1:]:
        if len(line) >= 1 and line[0] == '*':
            break

        if '%err' in line:
            errors = line[6:].split(';')

            if len(errors) >= 2:
                return [(x.split('=')[0].strip(), x.split('=')[1].strip()) for x in errors]


    return []

#%%
# Function to get the lines that include the child talking (one utterance)
# Needed because some utterances span multiple transcript lines
# @lines: the transcript split by newline
# @i: the first index of the child talking
# @return: a list of transcript lines making up the utterance
def get_token_lines(lines, i):
    token_lines = [lines[i]]

    for line in lines[i+1:]:
        if len(line) >= 1 and (line[0] == '%' or line[0] == '*'):
            break
        else:
            token_lines.append(line)
    
    return token_lines

#%%
# Tokenize one utterance of the child's speech
# @line: a list containing the utterance
# @errors: a list of tuples containing the errors and corresponding fixes for this utterance
# @return: the tokens for this line, cleaned and tokenized by NLTK
def get_tokens(lines, errors):
    pattern = re.compile(r'(\*CHI[0-9]?:|\[[^\[\]]*\]|\([^\[\)]*\)|\x15[0-9]+_[0-9]+\x15|[<>]|\+\.+)')
    tokenized_lines = []

    for line in lines:
        cleaned_line = re.sub(pattern, '', line).strip()
        
        for error in errors:
            cleaned_line = re.sub(error[0], error[1], cleaned_line)
        
        tokenized_lines += [x for x in nltk.word_tokenize(cleaned_line) if x.isalpha()]

    return tokenized_lines

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
            lines = data.split('\n')
            participants = get_target_participants(data)

            for participant in participants:
                tokens = []
                tags = []
                num_tokens = 0
                num_utterances = 0

                for i, line in enumerate(lines):
                    if f'*{participant[0]}:' in line:
                        tag_lines = get_tag_lines(lines, i)
                        tags += get_tags(tag_lines)

                        errors = get_errors(lines, i)
                        
                        token_lines = get_token_lines(lines, i)
                        tokens += get_tokens(token_lines, errors)

                        num_utterances += 1

                stage = get_stage(path)

                stats.append((path.split('/')[-1],
                              child_class,
                              participant,
                              stage,
                              tokens,
                              Counter(tokens),
                              Counter(tags),
                              len(tokens),
                              len(tags),
                              len(tokens)/num_utterances))

    df = pd.DataFrame(stats, columns=['file_name','child_class', 'name', 'stage', 'tokens', 'types', 'tags', 'num_tokens', 'num_tags', 'mean_utterance_length'])
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
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


#%%
def main():
    make_output_dirs()

    df = get_stats()

    df.to_csv(OUTPUT_PATH)

if __name__ == "__main__":
    main()
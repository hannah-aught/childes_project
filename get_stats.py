#%%
import nltk
import glob
import os
import re
from collections import defaultdict, Counter
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
nltk.download('punkt')

#%%
PATHS = glob.glob('./data/*/*/*.cha')
PATHS.sort()
OUTPUT_DIR = './output/'
OUTPUT_PATH = OUTPUT_DIR + 'language_data.csv'
PLOTS_DIR = OUTPUT_DIR + 'plots/'
TABLE_DIR = OUTPUT_DIR + 'tables/'

#%%
# function to get the participants from a transcript file
# @text: the text from the file, NOT split by lines
# @return: a list of tuples, each containing the name of the child,
# and the corresponding line label used in the transcript
def get_target_participants(text):
    participants = []
    # regex pattern with groups to match the child tag and child pseudonym
    pattern = re.compile(r'(CHI[0-9]?) ([a-zA-Z]+) Target_Child')
    matches = pattern.finditer(text)

    # Add the child tag and their name to the participants list as a tuple
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
    
    # loop over lines and look for %mor line for this utterance
    for j, line in enumerate(lines[i+1:]):
        # next utterance, break
        if len(line) >= 1 and line[0] == '*':
            break
        
        if '%mor' in line:
            # once we find a tag line, append the first one, then see if it spans multiple lines
            tag_lines.append(line)

            # loop over the next lines to see if there's more
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
    # regex for what to remove from the tag lines to be left with only tags
    pattern = re.compile(r'(%mor:\t|\|[^\s~]+\s*|:\w+|~|\+\.+)')

    # loop over the lines and remove the anything matching regex above from each to get tags
    for line in lines:
        tags += [x for x in re.sub(pattern, ' ', line).split() if x.isalpha()]

    return tags

#%%
# Function to get the errors noted in for an utterance
# @lines: the transcript split by newline
# @i: the first index of the child talking
# @return: a list of tuples containing the errors and fixes in the form (error, fix)
def get_errors(lines, i):
    # loop over the lines after where we found the start of the utterance and look for an %err line
    for line in lines[i+1:]:
        # if we've started a new utterance, stop looking
        if len(line) >= 1 and line[0] == '*':
            break
        
        if '%err' in line:
            # remove %err:\t from the start
            errors = line[6:].split(';')

            # if length < 2, there are no corrections
            if len(errors) >= 2:
                return [(x.split('=')[0].strip(), x.split('=')[1].strip()) for x in errors]

    # we found none if we didn't return in the loop
    return []

#%%
# Function to get the lines that include the child talking (one utterance)
# Needed because some utterances span multiple transcript lines
# @lines: the transcript split by newline
# @i: the first index of the child talking
# @return: a list of transcript lines making up the utterance
def get_token_lines(lines, i):
    # the first line will always be included because we found it in the calling function
    token_lines = [lines[i]]

    # loop over the next lines to see if this utterance spans multiple lines
    for line in lines[i+1:]:
        # if the next line begins with % or *, it's the start of a new utterance or of tags/errors/comments
        if len(line) >= 1 and (line[0] == '%' or line[0] == '*'):
            break
        # otherwise it's part of the utterance, add to token lines
        else:
            token_lines.append(line)
    
    return token_lines

#%%
# Tokenize one utterance of the child's speech
# @line: a list containing the utterance
# @errors: a list of tuples containing the errors and corresponding fixes for this utterance
# @return: the tokens for this line, cleaned and tokenized by NLTK
def get_tokens(lines, errors):
    # regex pattern for what to remove for data cleaning
    pattern = re.compile(r'(\*CHI[0-9]?:|\[[^\[\]]*\]|\([^\[\)]*\)|\x15[0-9]+_[0-9]+\x15|[<>]|\+\.+)')
    tokenized_lines = []

    # loop over lines and clean / tokenize
    for line in lines:
        # remove what we don't want
        cleaned_line = re.sub(pattern, '', line).strip()
        
        # replace all errors with their corrections
        for error in errors:
            cleaned_line = re.sub(error[0], error[1], cleaned_line)
        
        # tokenize the line using nltk and remove punctuation
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
    # make a list to hold tuples of features from each file. We'll use this to build the dataframe
    stats = []

    # loop over each path and extract our features
    for path in PATHS:
        with open(path, 'r') as f:
            # the child class is indicated by which folder inside data it's in
            child_class = path.split('/')[2]
            # read the whole file
            data = f.read()
            # get the individual lines (could have done readlines, but we need the whole text too)
            lines = data.split('\n')
            # get the participants for this file
            participants = get_target_participants(data)

            # loop over the participants and collect features for each of them
            for participant in participants:
                # initialize collector variables
                tokens = []
                tags = []
                num_tokens = 0
                num_utterances = 0
                stage = get_stage(path)

                # loop over the lines
                for i, line in enumerate(lines):
                    # when we fine a line with our participant, get the tokens and tags
                    if f'*{participant[0]}:' in line:
                        # get the lines containing tag data and extract tags
                        tag_lines = get_tag_lines(lines, i)
                        tags += get_tags(tag_lines)

                        # get any corresponding errors and corrections
                        errors = get_errors(lines, i)
                        
                        # get the lines containing tokens and clean / tokenize
                        token_lines = get_token_lines(lines, i)
                        tokens += get_tokens(token_lines, errors)

                        # increment for each utterance we see
                        num_utterances += 1

                # append the tuple with our data for this participant
                stats.append((path.split('/')[-1],
                              child_class,
                              participant,
                              stage,
                              tokens,
                              Counter(tokens),
                              Counter(tags),
                              len(tokens),
                              len(Counter(tokens)),
                              len(Counter(tags)),
                              len(tokens)/num_utterances))
    
    # make the dataframe from stats and return it
    df = pd.DataFrame(stats, columns=['file_name','child_class', 'child_name', 'stage', 'tokens','types', 'tags', 'num_tokens', 'num_types', 'num_tags', 'mlu'])
    return df

#%%
# Function to make a barplot of data
# @df: dataframe containing the data to graph
# @ax: a matplotlib axis to plot onto
# @y: label for the feature to graph
# @title: title for the graph
# @ylabel: string with the label for the y axis of the graph
def make_plot(df, y, ax, title, ylabel):
    # make a barplot of the given feature on $ax
    sns.barplot(data=df, x='stage', y=y, hue='child_class', palette='husl', capsize=0.1, ax=ax)
    # set the axis labels and title and remove the legend
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Brown Development Stage")
    ax.legend().remove()
    ax.set_title(title)

#%%
# function to generate the development curves from the collected data
# @data: an array of the stat dicts for monolingual and bilingual data
# @path: the path to write the plots to
def generate_plots(df):
    # make a figure with 4 subplots for our 4 plots
    f, axs = plt.subplots(2, 2, sharex='row')
    # lists with the feature names and y labels for each
    df_cols = ['num_tokens', 'num_tags', 'num_types', 'mlu']
    ylabels = ['Vocabulary Size', 'Number of Tag Types', 'Number of Tokens', 'Mean Utterance Length']

    # loop over the 4 features and make each plot
    for i in range(4):
        make_plot(df, df_cols[i], axs[i//2, i%2], f'Figure 3.{i+1}', ylabels[i])
    
    # we only want one legend for the figure and we want it in the upper left
    axs[0,0].legend(loc='upper left')
    # tight layout so text doesn't overlap
    f.tight_layout()
    # savee the figure
    f.savefig(PLOTS_DIR + 'barplots.png')

#%%
# Function to aggregate data in the dataframe over a specific column
# Used to aggregate data for saving/reporting in tables
# @df: the dataframe
# @aggregate_col: the name of the column to aggregate on
# @col_name: the new name for this column
# @return: a new dataframe with all columns but class, stage, and $aggregate_col dropped
# and aggregate_col renamed to col_name
def aggregate_along_col(df, feature, col_name):
    # group by class and stage, get the mean and std for the feature
    aggregated = df[['child_class', 'stage', feature]].groupby(['child_class', 'stage']).agg({feature:['mean', 'std']})
    # rename the index and columns for a prettier table
    aggregated.index = aggregated.index.rename(['Language Group', 'Stage'])
    aggregated.columns = [f'Mean {col_name}', 'Standard Deviation']
    return aggregated

#%%
# Function to aggregate the type and tag data across each of the three files per child/stage that we have
# @df: the dataframe created using get_stats
# @return: a new dataframe without file_name or tokens and with the following aggregations:
# types, tags, and num_tokens: summed across the three files per child/stage
# mean_utterance_length: mean of the mean lengths from the three files per child/stage 
# num_types, num_tags: the length of the new counter object
def get_aggregate_data(df):
    # drop the file name, tokens, num_types, and num_tags because we're not using the first two and recalculating the second two
    adf = df.drop(columns=['file_name','tokens','num_types','num_tags'], inplace=False)
    # group the new dataframe by language group, name, and development stage
    adf = adf.groupby(['child_class','child_name','stage'])
    # combine the statistics for each child in each stage
    adf = adf.agg({'types':'sum','tags':'sum','num_tokens':'sum','mlu':'mean'})
    # get the new number of tags and types for each stage
    adf['num_tags'] = adf['tags'].apply(len)
    adf['num_types'] = adf['types'].apply(len)
    # get rid of multiindexing from grouping
    adf.reset_index(inplace=True)
    return adf

#%%
# Function to run significance testing on the results (independent t-test)
# @df: dataframe containing the statistics collected from the corpus files (not aggregated)
# @return: a dataframe containing the t-statistics and p-values for the features we chose to explore
def test_significance(df):
    # These are the features we'll be looking at
    features = ['num_tokens', 'num_types', 'num_tags', 'mlu']
    stages = [1, 3, 5]

    # List we'll fill with tuples to build dataframe later
    t_tests = []

    # make one df for the bilingual data and one for monolingual to compare
    bi_df = df[df['child_class'] == 'Bilingual']
    mono_df = df[df['child_class'] == 'Monolingual']
    
    # loop over the features and run the t-test for each
    for feature in features:
        # also loop over the stages so we're getting the significance for each stage
        for stage in stages:
            # get the correct feature/stage data from both dataframes
            bi_data = bi_df[bi_df['stage'] == stage][feature]
            mono_data = mono_df[mono_df['stage'] == stage][feature]
            # run the t-test and add it to the t_test list
            t_stat, p_val = stats.ttest_ind(bi_data, mono_data)
            t_tests.append((feature, stage, t_stat, p_val))
    
    # make and return a new dataframe
    test_df = pd.DataFrame(t_tests, columns=['statistic', 'stage', 't-statistic', 'p-val'])
    return test_df
        

#%%
# Function to make the output directories for monolingual and bilingual data
# if they don't already exist
# @return:
def make_output_dirs():
    # check if each dir exists, then make it if not
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

#%%
def main():
    # make the output, plot, and table dirs
    make_output_dirs()

    # parse the dataset and collect the relevant features
    df = get_stats()

    # perform independent t-tests on our four features across both groups and all stages
    significance_df = test_significance(df)

    # aggregate the data by language group and development stage
    aggregated_df = get_aggregate_data(df)

    # generate the plots using the aggregated data
    generate_plots(aggregated_df)

    # generage tables for all our features
    vocab_df = aggregate_along_col(df, 'num_types', 'Vocab Size')
    tag_df = aggregate_along_col(df, 'num_tags', 'POS')
    token_df = aggregate_along_col(df, 'num_tokens', 'Tokens')
    mlu_df = aggregate_along_col(df, 'mlu', 'Utterance Length')

    # save the tables as .tex files so we can have pretty tables
    vocab_df.to_latex(TABLE_DIR + 'vocab.tex')
    tag_df.to_latex(TABLE_DIR + 'tags.tex')
    token_df.to_latex(TABLE_DIR + 'tokens.tex')
    mlu_df.to_latex(TABLE_DIR + 'mlu.tex')
    significance_df.to_latex(TABLE_DIR + 'significance.tex')

    # save the features, aggregated features, and significance results to csv files
    df.to_csv(OUTPUT_DIR + 'language_data.csv')
    aggregated_df.to_csv(OUTPUT_DIR + 'aggregated_language_data.csv')
    significance_df.to_csv(OUTPUT_DIR + 'significance_tests.csv')

if __name__ == "__main__":
    main()
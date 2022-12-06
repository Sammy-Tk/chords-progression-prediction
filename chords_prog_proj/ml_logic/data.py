# from chords_prog_proj.ml_logic.params import (COLUMN_NAMES_RAW,
#                                             DTYPES_RAW_OPTIMIZED,
#                                             DTYPES_RAW_OPTIMIZED_HEADLESS,
#                                             DTYPES_PROCESSED_OPTIMIZED
#                                             )

# from chords_prog_proj.data_sources.local_disk import (get_pandas_chunk, save_local_chunk)

# from chords_prog_proj.data_sources.big_query import (get_bq_chunk, save_bq_chunk)

import os
import pandas as pd
import string
import re
import numpy as np
from itertools import groupby

from chords_prog_proj.ml_logic.params import DATA_FILE_KAGGLE, DATA_FILE_JAZZ


'''
GET LOCAL DATA
'''
def get_csv_data():
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'mlops/data/raw', DATA_FILE_KAGGLE)
    raw_csv_df = pd.read_csv(data_path)
    return raw_csv_df

def get_text_data():
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'mlops/data/raw', DATA_FILE_JAZZ)
    raw_txt_df = pd.read_csv(data_path, sep="_START_|_END_", header=None, engine='python').T
    return raw_txt_df


'''
TURN SINGLE CHORDS COLUMN TO DATAFRAME TO CONCAT
'''
def expand_cols(raw_txt_df):
    raw_txt_df.rename(columns={0: "chords"}, inplace=True)
    raw_txt_df.replace(' ', np.nan, inplace=True)
    raw_txt_df.dropna(inplace=True)
    raw_txt_df.insert(0, 'artist_name', 'unknown', True)
    raw_txt_df.insert(0, 'genres', 'jazz', True)
    raw_txt_df['song_name'] = 'unknown'
    return raw_txt_df


'''
DROP DUPLICATES BASED ON SONG & ARTIST TITLES
DROP UNWANTED COLUMNS
'''
def drop_cols(raw_df):

    nonrepeated_songs_df = raw_df.drop_duplicates(
                                subset=['artist_name', 'song_name'],
                                keep = 'first').reset_index(drop = True)

    slim_df = nonrepeated_songs_df.loc[:, ['artist_name', 'genres',
                                           'chords', 'song_name']]
    return slim_df


'''
CLEANING
'''
#replace useful symbols with strings
#delete useless symbols (break or just deletions)
def dashes_commas_colons(song):
    # Fix dashes, colons, and commas
    for idx, chord in enumerate(song):
        if '--' in chord:
            song[idx] = chord.split('--')[0]
        elif '-' in chord:
            linechords = chord.split('-')
            song[idx] = linechords[0]
            i = idx + 1
            for lc in linechords[1:]:
                song.insert(i, lc)
                i += 1
        elif ',' in chord:
            linechords = chord.split(',')
            song[idx] = linechords[0]
            i = idx + 1
            for lc in linechords[1:]:
                song.insert(i, lc)
                i += 1

    return song


# break on slash chords
def slashes(song):
    # Break all slash chords
    for idx, chord in enumerate(song):
        if '/' in chord:
            song[idx] = chord.split('/')[0]
        elif '|' in chord:
            song[idx] = chord.split('|')[0]
        elif '\\' in chord:
            song[idx] = chord.split('\\')[0]

    return song

# translate useful symbols to strings so we don't accidentally delete them
def translations(song):
    #
    useful_symbols = {'*': 'dim', '°': 'dim', 'º': 'dim', 'o': 'dim',
                    '+': 'aug', '#': 'sharp', ':': '', 'flat': 'b'}

    for idx, chord in enumerate(song):
        for sym in useful_symbols:
            if sym in chord:
                song[idx] = chord.replace(sym, useful_symbols[sym])

    return song

# delete unhelpful symbols
def punctuation(song):

    cleaned_song = song.copy()

    cleaned_song = dashes_commas_colons(cleaned_song)

    cleaned_song = slashes(cleaned_song)

    cleaned_song = translations(cleaned_song)

    # delete all other symbols
    for idx, chord in enumerate(cleaned_song):
        cleaned_song[idx] = re.sub(r'[^\w\s]', '', chord)

    #translate sharps back into symbols
    for idx, chord in enumerate(cleaned_song):
        if 'sharp' in cleaned_song[idx]:
            cleaned_song[idx] = chord.replace('sharp', '#')

    return cleaned_song

# merge chords into single format
# delete non-chords (words)
def merge_chords(song):
    # dictionary of correct formats (keys) and incorrect (values)
    chords_format = {'M7': ['major7', 'maj7', '7M', 'Major7', 'Maj7'],
                    'm7': ['minor7', 'min7'],
                    'hdim7': ['h7', 'hdim', 'hdim7', 'h'],
                    '7': ['sus', '9', '11', '13'],
                    'aug': ['augmented'],
                    'm': ['minor', 'min'],
                    'dim': ['diminished'],
                    '': ['add', 'major', 'maj', 'M', 'Major', 'Maj', '2', '4', '6'],
                    }

    merged_song = song.copy()

    # substitute correct formats
    for key, value in chords_format.items():
        for i in value:
            for idx, chord in enumerate(merged_song):
                if i in chord:
                    merged_song[idx] = chord.split(i)[0] + key

    # delete words (based on second letter)
    for idx, chord in enumerate(merged_song):
        if len(chord) > 1:
            if chord[1] != '#' or chord[1] != 'b':
                if chord[1:] not in chords_format.keys():
                    del merged_song[idx]
            elif chord[1] == '#' or chord[1] == 'b':
                if chord[2:] not in chords_format.keys():
                    del merged_song[idx]

    return merged_song

'''
PARENT FUNCTION FOR CLEANING
'''
# convert strings to lists of strings
# filter chords by allowable first letter
# delete special words
# send to punctuation and merge
# remove consecutively repeating chords
# delete empty chords
# delete stand-alone numbers
def clean_chords(chords_column):

    letters = list(string.ascii_uppercase)[:7]
    cleaned = []

    column = chords_column.copy()

    for row in column:
        # Convert string to list of strings
        if type(row) is str:
            song_list = row.split()
        else:
            print('error: data in row not string;', f'{type(row)}')

        # Only chords that begin with designated letters
        raw_chords = [chord for chord in song_list if chord[0] in letters]

        # Delete 'chords' and 'chorus'
        for idx, chord in enumerate(raw_chords):
            if 'chor' in chord.lower() or 'bass' in chord.lower():
                del raw_chords[idx]

        # remove symbols
        unsymboled_chords = punctuation(raw_chords)

        # merge chords into same format
        merged_chords = merge_chords(unsymboled_chords)

        # Remove repeated chords
        non_repeating_chords = [c[0] for c in groupby(merged_chords)]

        # Delete empty strings and numbers
        clean_song = [chord for chord in non_repeating_chords if len(chord) > 0]
        clean_song = [chord for chord in clean_song if chord[0] in letters]

        cleaned.append(clean_song)

    return cleaned

'''
CONCAT CHORDS INTO ONE LONG STRING PER SONG
MAKE SONG TITLES ALL CAPS
'''
def new_columns(cleaned_df):
    new_columns_df = cleaned_df.copy()

    new_columns_df['chords_list'] = \
        [''.join(map(str, l)) for l in new_columns_df['chords']]
    new_columns_df['song_name'] = \
        [name.upper() for name in new_columns_df['song_name']]

    return new_columns_df



def song_length(clean_genres_df):
    song_len_df = clean_genres_df.copy()
    # new column
    song_len_df['song_length'] = 0

    for index in song_len_df.index:
        song_len_df.loc[index, 'song_length'] = len(song_len_df.loc[index, 'chords'])

    return song_len_df


'''
TRANSLATE MULTIPLE SUBGENRES TO MAIN GENRES (FOR MOST)
'''
def genre_cleaning(genres_column):
    # most common genres list
    popular_genres = ['pop', 'rock', 'jazz', 'folk', 'blues', 'country', 'world',
                    'sertanejo', 'adoracao', 'rock-and-roll', 'bossa nova', 'reggae',
                    'lounge', 'metal', 'pagode', 'latin worship', 'mpb']

    cleaned_genres = genres_column.copy()

    # replace lists of subgenres with main genre (in most but not all)
    for idx, element in enumerate(cleaned_genres):
        for genre in popular_genres:
            if genre in element:
                cleaned_genres[idx] = genre
            else:
                pass

    # change empty genres to 'unknown'
    for idx, element in enumerate(cleaned_genres):
        if element == '[]':
            cleaned_genres[idx] = 'unknown'
        else:
            pass

    return cleaned_genres

'''
FILTER DF BY CHOOSING ONLY SONGS LONGER THAN n CHORDS
'''
def filter_length(song_len_df, min_length):
    filt_len = song_len_df['song_length'] >= min_length
    kaggle_data_v1 = song_len_df.copy()[filt_len]

    return kaggle_data_v1


'''
SEND CLEAN DF TO CVS
'''
def df_to_csv(final_df, ts, save_path):

    filename = 'final_df' + '_' + ts + '.csv'
    # how do i get an object to the a string of it's name?
    my_path = os.path.join(save_path, filename)

    final_df.to_csv(my_path)

    return print(f'{filename} saved to {save_path}')

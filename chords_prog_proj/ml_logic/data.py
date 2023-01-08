# from chords_prog_proj.data_sources.local_disk import (get_pandas_chunk, save_local_chunk)
# from chords_prog_proj.data_sources.big_query import (get_bq_chunk, save_bq_chunk)

import os
import pandas as pd
import string
import re
import numpy as np
from itertools import groupby

from colorama import Fore, Style

from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import urllib.request

from chords_prog_proj.ml_logic.params import (LOCAL_DATA_PATH, DATA_FILE_KAGGLE_RAW, DATA_FILE_LSTM_REALBOOK_RAW)

def get_data_kaggle():
    """Get csv file"""

    root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, LOCAL_DATA_PATH, 'raw', DATA_FILE_KAGGLE_RAW)

    # If the raw file does not exist, download it from Kaggle.
    # This requires an API key to authenticate to Kaggle.
    # To set up your API Key: go to your Kaggle account Tab at https://www.kaggle.com/<username>/account
    # click ‘Create API Token’. A file named kaggle.json will be downloaded - move this file to ~/.kaggle/
    if not os.path.exists(data_path):
        print(Fore.BLUE + "\nDownloading csv file from Kaggle..." + Style.RESET_ALL)
        api = KaggleApi()
        api.authenticate()
        # Download file chords_and_lyrics.csv.zip
        api.dataset_download_file('eitanbentora/chords-and-lyrics-dataset',
                                file_name='chords_and_lyrics.csv')

        # Extract zip file into the directory "data/raw"
        with zipfile.ZipFile("chords_and_lyrics.csv.zip", "r") as zipref:
            zipref.extractall(path=os.path.join(root_path, LOCAL_DATA_PATH, 'raw'))

        # Rename extracted csv file
        os.rename(os.path.join(root_path, LOCAL_DATA_PATH, "raw", "chords_and_lyrics.csv"),
                os.path.join(root_path, LOCAL_DATA_PATH, "raw", DATA_FILE_KAGGLE_RAW))

        # Delete zip file
        if os.path.exists("chords_and_lyrics.csv.zip"):
            os.remove("chords_and_lyrics.csv.zip")
        else:
            print("The zip file does not exist")

    raw_csv_df = pd.read_csv(data_path)
    return raw_csv_df

def get_data_lstm_realbook():
    """Get txt file"""

    root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, LOCAL_DATA_PATH, 'raw', DATA_FILE_LSTM_REALBOOK_RAW)

    # If the raw file does not exist, download it
    if not os.path.exists(data_path):
        print(Fore.BLUE + "\nDownloading txt file LSTM Realbook..." + Style.RESET_ALL)
        url="https://raw.githubusercontent.com/keunwoochoi/lstm_real_book/master/chord_sentences.txt"
        urllib.request.urlretrieve(url, data_path)

    raw_txt_df = pd.read_csv(data_path, sep="_START_|_END_", header=None, engine='python').T
    return raw_txt_df


def expand_cols(raw_txt_df):
    """Turn txt file with a single column that contains chords, to a dataframe """

    raw_txt_df.rename(columns={0: "chords"}, inplace=True)
    raw_txt_df.replace(' ', np.nan, inplace=True)
    raw_txt_df.dropna(inplace=True)
    raw_txt_df.insert(loc=0, column='song_name', value='unknown', allow_duplicates=True)
    raw_txt_df.insert(loc=0, column='artist_name', value='unknown', allow_duplicates=True)
    raw_txt_df.insert(loc=0, column='genres', value='jazz', allow_duplicates=True)
    raw_txt_df.reset_index(drop=True, inplace=True)

    return raw_txt_df


def drop_cols(df_raw):
    """Drop duplicates based artist id and song title, and remove unwanted columns"""

    df_nonrepeated_songs = df_raw.drop_duplicates(
                                subset=['artist_id', 'song_name'],
                                keep = 'first').reset_index(drop = True)

    df_slim = df_nonrepeated_songs.loc[:, ['genres', 'artist_name','song_name', 'chords']]

    return df_slim


def remove_guitar_tabs(df):
    """Remove songs that contain guitar tabs
    e.g. 'B|-----1-----------1---------1-----|'
    """
    # We need the escape character "\|", because "|" means OR
    character_guitar_tab = "\|-"

    # Define boolean filter, and take its inverse with ~
    filt = ~df['chords'].str.contains(character_guitar_tab)
    # Apply boolean filter to dataFrame
    df_without_tabs = df[filt].reset_index()

    return df_without_tabs


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


def clean_chords(chords_column):
    '''
    Parent function for cleaning:
        Convert strings to lists of strings
        Filter chords by allowable first letter
        Delete special words
        Send to punctuation and merge
        Remove consecutively repeating chords
        Delete empty chords
        Delete stand-alone numbers
    '''

    # Generate list ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    letters = list(string.ascii_uppercase)[:7]

    cleaned = []

    column = chords_column.copy()

    for row in column:
        if type(row) is str:
            # Remove single quotes (some chords are preceded by a single quote, e.g. 'A)
            row = row.replace("'", "")
            # Convert string to list of strings
            song_list = row.split()
        else:
            print('error: data in row not string;', f'{type(row)}')

        # Only chords that begin with designated letters
        raw_chords = [chord for chord in song_list if chord[0] in letters]

        # Delete 'chords' and 'chorus'
        for idx, chord in enumerate(raw_chords):
            if 'chor' in chord.lower() or 'bass' in chord.lower():
                del raw_chords[idx]

        # Remove symbols
        unsymboled_chords = punctuation(raw_chords)

        # Merge chords into same format
        merged_chords = merge_chords(unsymboled_chords)

        # Remove repeated chords
        non_repeating_chords = [c[0] for c in groupby(merged_chords)]

        # Delete empty strings and numbers
        clean_song = [chord for chord in non_repeating_chords if len(chord) > 0]
        clean_song = [chord for chord in clean_song if chord[0] in letters]

        cleaned.append(non_repeating_chords)

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

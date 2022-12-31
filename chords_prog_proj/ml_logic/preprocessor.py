import pandas as pd
import os
import time
from colorama import Fore, Style

from chords_prog_proj.ml_logic.data import (get_data_kaggle, get_data_lstm_realbook, expand_cols, \
            drop_cols, clean_chords, new_columns, song_length, genre_cleaning, \
            filter_length, df_to_csv)
from chords_prog_proj.ml_logic.utils import (count_chords, count_artists, count_genres)


def pre_clean():
    """Read data, pre-cleaning and concatenate"""

    # Get raw data
    df_kaggle_raw = get_data_kaggle()
    df_lstm_realbook_raw = get_data_lstm_realbook()

    # Select columns
    df_kaggle_selected_cols = drop_cols(df_kaggle_raw)
    df_lstm_realbook_selected_cols = expand_cols(df_lstm_realbook_raw)

    # Concatenate dataFrames
    df_concatenated = pd.concat([df_kaggle_selected_cols, df_lstm_realbook_selected_cols], ignore_index=True)

    print(Fore.GREEN + f'\n✅ Data read and merged, with {len(df_concatenated)} songs.' + Style.RESET_ALL)

    return df_concatenated


def clean(df_concatenated):
    df_cleaned = df_concatenated.copy()

    cleaned_chords_column = clean_chords(df_cleaned['chords'])
    df_cleaned['chords'] = cleaned_chords_column

    print(f'\n✅ Chords cleaned.')

    return df_cleaned


def preprocess(get_distributions=False):

    print(Fore.BLUE + "\n⭐️ Use case: preprocess" + Style.RESET_ALL)
    df_concatenated = pre_clean()

    print(Fore.BLUE + f'\n✅ Cleaning chords... may take up to 1 minute.' + Style.RESET_ALL)
    cleaned_df = clean(df_concatenated)

    # drop duplicates based on chords and song name
    new_columns_df = new_columns(cleaned_df)

    unreplicated_df = \
        new_columns_df.drop_duplicates(subset=['chords_list', 'song_name'],
                                    keep = 'last').reset_index(drop = True)

    # drop unnecessary columns
    unreplicated_df.drop(columns=['song_name', 'chords_list'], inplace=True)

    print(f'\n✅ Genre cleaning... may take up to 1 minute.')

    # clean genres
    slim_genres = genre_cleaning(unreplicated_df['genres'])
    clean_genres_df = unreplicated_df.copy()
    clean_genres_df['genres'] = slim_genres

    # get song length column, and filter by length
    song_len_df = song_length(clean_genres_df)
    final_df = filter_length(song_len_df, 8)

    print(f'{len(final_df)} of {len(concat_df)} songs kept during preprocessing.')

    if get_distributions == False:
        pass
    else:
        print(f'\n✅ Top five chords, grenres, and artists being generated.')
        chord_count_df = count_chords(final_df, low_freq_to_remove=10,
                                      histplot=True, ascending=False)
        print(chord_count_df.head(5))
        genre_count_df = count_genres(final_df, histplot=True)
        print(genre_count_df.head(5))
        artists_df = count_artists(final_df, histplot=True)
        print(artists_df.head(5))

    # send to csv
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_proc_folder = os.path.join(root_path, 'mlops/data/processed')

    ts = time.strftime("%d-%m-%y_%H:%M")
    df_to_csv(final_df, ts, data_proc_folder)

    print(f'\n✅ Preprocessor finished.')

    return None

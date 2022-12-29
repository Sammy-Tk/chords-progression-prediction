import pandas as pd
import os
import time

from chords_prog_proj.ml_logic.data import get_data_kaggle, get_data_lstm_realbook, expand_cols, \
            drop_cols, clean_chords, new_columns, song_length, genre_cleaning, \
            filter_length, df_to_csv
from chords_prog_proj.ml_logic.utils import count_chords, count_artists, count_genres


'''
READ DATA, PRE-CLEANING ENGINEERING AND CONCAT
'''
def pre_clean():
    # read data
    raw_kaggle_df = get_data_kaggle()
    raw_jazz_col = get_data_lstm_realbook()

    # merge
    raw_jazz_df = expand_cols(raw_jazz_col)
    slim_kaggle_df = drop_cols(raw_kaggle_df)
    concat_df = pd.concat([slim_kaggle_df, raw_jazz_df], ignore_index=True)

    print(f'\n✅ Data read and merged, with {len(concat_df)} songs.')

    return concat_df


'''
CLEAN
'''
def clean(concat_df):
    cleaned_df = concat_df.copy()

    cleaned_chords_column = clean_chords(cleaned_df['chords'])
    cleaned_df['chords'] = cleaned_chords_column

    print(f'\n✅ Chords cleaned.')

    return cleaned_df


'''
PREPROCESS
'''
def preprocess(get_distributions=False):

    print("\n⭐️ Use case: preprocess")

    concat_df = pre_clean()

    print(f'\n✅ Cleaning chords... may take up to 1 minute.')
    cleaned_df = clean(concat_df)

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

#EC

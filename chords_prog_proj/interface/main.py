import numpy as np
import pandas as pd
import ast
import os
import json

from colorama import Fore, Style

from tensorflow.keras.utils import to_categorical

from chords_prog_proj.ml_logic.data import clean_data, get_chunk, save_chunk
from chords_prog_proj.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from chords_prog_proj.ml_logic.params import CHUNK_SIZE, DATASET_SIZE, VALIDATION_DATASET_SIZE, LOCAL_DATA_PATH, DATA_FILE_KAGGLE, DATA_FILE_JAZZ
from chords_prog_proj.ml_logic.preprocessor import preprocess_features
from chords_prog_proj.ml_logic.utils import get_dataset_timestamp
from chords_prog_proj.ml_logic.registry import get_model_version

from chords_prog_proj.ml_logic.registry import load_model, save_model


def preprocess(source_type = 'train'):
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    - source_type: 'train' or 'val'
    """

    print("\n⭐️ Use case: preprocess")

    # Iterate on the dataset, in chunks
    chunk_id = 0
    row_count = 0
    cleaned_row_count = 0
    source_name = f"{source_type}_{DATASET_SIZE}"
    destination_name = f"{source_type}_processed_{DATASET_SIZE}"

    while (True):
        print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

        data_chunk = get_chunk(
            source_name=source_name,
            index=chunk_id * CHUNK_SIZE,
            chunk_size=CHUNK_SIZE
        )

        # Break out of while loop if data is none
        if data_chunk is None:
            print(Fore.BLUE + "\nNo data in latest chunk..." + Style.RESET_ALL)
            break

        row_count += data_chunk.shape[0]

        data_chunk_cleaned = clean_data(data_chunk)

        cleaned_row_count += len(data_chunk_cleaned)

        # Break out of while loop if cleaning removed all rows
        if len(data_chunk_cleaned) == 0:
            print(Fore.BLUE + "\nNo cleaned data in latest chunk..." + Style.RESET_ALL)
            break

        X_chunk = data_chunk_cleaned.drop("fare_amount", axis=1)
        y_chunk = data_chunk_cleaned[["fare_amount"]]

        X_processed_chunk = preprocess_features(X_chunk)

        data_processed_chunk = pd.DataFrame(
            np.concatenate((X_processed_chunk, y_chunk), axis=1)
        )

        # Save and append the chunk
        is_first = chunk_id == 0

        save_chunk(
            destination_name=destination_name,
            is_first=is_first,
            data=data_processed_chunk
        )

        chunk_id += 1

    if row_count == 0:
        print("\n✅ No new data for the preprocessing 👌")
        return None

    print(f"\n✅ Data processed saved entirely: {row_count} rows ({cleaned_row_count} cleaned)")

    return None

def train():
    """
    Train a new model on the full (already preprocessed) dataset.
    Save final model once it has seen all data, and compute validation metrics on a holdout validation set.
    """
    print("\n⭐️ Use case: train")

    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Important parameters
    # function get_X_y
    length = 12
    # function create_dataset
    number_of_samples = 500000

    def csv_to_concat_df(csv_1, csv_2):
        df_1 = pd.read_csv(csv_1)
        df_2 = pd.read_csv(csv_2)
        return pd.concat([df_1, df_2],axis=0,ignore_index=True)

    path_csv_1 = os.path.join(os.getcwd(), LOCAL_DATA_PATH, "processed", DATA_FILE_KAGGLE)
    path_csv_2 = os.path.join(os.getcwd(), LOCAL_DATA_PATH, "processed", DATA_FILE_JAZZ)

    df = csv_to_concat_df(path_csv_1, path_csv_2)

    # Convert strings to lists
    df['chords'] = df['chords'].apply(ast.literal_eval)

    X = df['chords']

    # Create a dictionary which stores a unique token for each chord:
    # the key is the chord while the value is the corresponding token.
    chord_to_id = {}
    chord_to_id['UNKNOWN'] = 0
    iter_ = 1
    for song in X:
        for chord in song:
            if chord in chord_to_id:
                continue
            chord_to_id[chord] = iter_
            iter_ += 1

    # Save dictionary to json file
    with open('chord_to_id.json', 'w') as f:
        json.dump(chord_to_id, f)

    # Vocab size
    vocab_size = len(chord_to_id)

    def get_X_y(song, length):
        """
        Function that, given a song (list of chords), returns:
            - a list of strings (list of chords) that corresponds to part of the song - this string should be of length N
            - the chord that follows the previous list of chords
        """
        if len(song) <= length:
            return None

        first_chord_idx = np.random.randint(0, len(song)-length)

        X_chords = song[first_chord_idx:first_chord_idx+length]
        y_chords = song[first_chord_idx+length]

        return X_chords, y_chords

    def create_dataset(songs, number_of_samples):
        """
        Function, that, based on the previous function and the loaded songs, generate a dataset X and y:
        - each sample of X is a list of chords
        - the corresponding y is the chord that comes just after in the input list of chords
        """

        X, y = [], []
        indices = np.random.randint(0, len(songs), size=number_of_samples)

        for idx in indices:
            ret = get_X_y(songs[idx], length=length)
            if ret is None:
                continue
            xi, yi = ret
            X.append(xi)
            y.append(yi)

        return X, y

    # Create X and y
    X, y = create_dataset(X, number_of_samples)

    # Split X and y in train and test data.
    len_ = int(0.7*len(X))
    chords_train = X[:len_]
    chords_test = X[len_:]
    y_train = y[:len_]
    y_test = y[len_:]

    # Based on the dictionary of chords to id, tokenize X and y (chords to numbers)
    X_train = [[chord_to_id[chord] for chord in song] for song in chords_train]
    X_test = [[chord_to_id[chord] if chord in chord_to_id else chord_to_id['UNKNOWN'] for chord in song ] for song in chords_test]
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    y_train_token = [chord_to_id[chord] for chord in y_train]
    y_test_token = [chord_to_id[chord] if chord in chord_to_id else chord_to_id['UNKNOWN'] for chord in y_test]

    # Convert the tokenized outputs to one-hot encoded categories
    y_train_cat = to_categorical(y_train_token, num_classes=vocab_size)
    y_test_cat = to_categorical(y_test_token, num_classes=vocab_size)

    # Load model
    model = None
    model = load_model()  # production model
    # model = load_model('model_v1')

    # Model params
    learning_rate = 0.001
    epochs = 2
    batch_size = 256
    patience = 15
    output_dim = 50

    # Initialize model
    if model is None:
        model = initialize_model(vocab_size=vocab_size, output_dim=output_dim)

    print(Fore.BLUE + "\nCompile and fit preprocessed validation data..." + Style.RESET_ALL)

    # Compile and train the model
    model = compile_model(model, learning_rate)
    model, history = train_model(
        model,
        X_train,
        y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )

    # Return the highest value of the validation accuracy
    metrics_val = np.max(history.history['val_accuracy'])
    print(f"\n✅ Trained with accuracy: {round(metrics_val, 2)}")

    params = dict(
        # Model parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,

        # Package behavior
        context="train",

        # Data source
        model_version=get_model_version(),
    )

    # Save model
    save_model(model=model, params=params, metrics=dict(accuracy=metrics_val))

    # Evaluate on the test set
    print(Fore.BLUE + "\nEvaluate on the test set..." + Style.RESET_ALL)
    metrics_dict = evaluate_model(model=model, X=X_test, y=y_test_cat)
    #accuracy = metrics_dict["accuracy"]

    return metrics_val


def pred(song: list = None,
         n_chords=4) -> str:
    """
    Make a prediction using the latest trained model
    """
    print("\n⭐️ Use case: predict")

    # Load dictionary chords_to_id
    with open("chord_to_id.json", "r") as json_file:
        chord_to_id = json.load(json_file)

    # Create dictionary id_to_chord
    id_to_chord = {v: k for k, v in chord_to_id.items()}

    model = load_model()

    if song is None:
        song = ['G', 'D', 'G', 'D', 'Am', 'F', 'Em', 'F#',]

    def get_predicted_chord(song):
        # Convert chords to numbers (tokens)
        song_convert = [chord_to_id[chord] for chord in song]

        # Return an array of size vocab_size, with the probabilities
        pred = model.predict([song_convert], verbose=0)
        # Return the index of the highest probability
        pred_class = np.argmax(pred[0])
        # Turn the index into a chord
        pred_chord = id_to_chord[pred_class]

        return pred_chord

    def repeat_prediction(song, repetition):
        song_tmp = song
        for i in range(repetition):
            predicted_chord = get_predicted_chord(song_tmp)
            song_tmp.append(predicted_chord)
            #song_tmp = song_tmp[1:]

        return song_tmp

    # Return the chord with (n)th probability
    def output_nth_chord_pred(n):
        return id_to_chord[np.argsort(np.max(pred, axis=0))[-n]]

    # Return dictionary of the chords with n(th) probability
    def outputs_next_chord(chords, n):
        next_chord_dict = {}
        for i in range(n):
            next_chord_dict[f"{i+1}th_pred_chord"] = output_nth_chord_pred(i+1)
        return next_chord_dict

    chord_pred = get_predicted_chord(song)
    print("\n✅ predicted next chord: ", chord_pred)

    n_chords_pred = repeat_prediction(song, n_chords)
    print(f"\n✅ predicted {n_chords} next chords: ", n_chords_pred)

    outputs_next_chords = outputs_next_chord()

    return chord_pred


if __name__ == '__main__':
    # preprocess()
    # preprocess(source_type='val')
    train()
    song = ['Cm', 'Bb', 'Ab', 'G7', 'Cm', 'Bb', 'Ab', 'G7']
    pred(song=song, n_chords=10)
    # evaluate()
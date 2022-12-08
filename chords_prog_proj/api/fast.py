
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chords_prog_proj.interface.main import pred
from chords_prog_proj.ml_logic.registry import load_model

import random
import numpy as np
import json

app = FastAPI()
# To avoid loading the model at each GET request:
# Load the model into memory on startup and store it in a global variable in app.state,
# which is kept in memory and accessible across all routes
app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/predict_baseline")
def predict_baseline(chords: str):
    i = random.randint(0,1)
    if i == 0:
        return {'The next chord is': 'C'}
    else:
        return {'The next chord is': 'G'}


@app.get("/predict")
def predict(song, n_chords, randomness):
    """
    Make a prediction using the latest trained model
    """
    song = song.split(',')
    n_chords = int(n_chords)
    randomness = int(randomness)

    n_chords_pred = pred(song=song, n_chords=n_chords, randomness=randomness, model=app.state.model)

    return {'predicted_chord': n_chords_pred}


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END

# @app.get("/predict")
# def predict(pickup_datetime: datetime,  # 2013-07-06 17:18:00
#             pickup_longitude: float,    # -73.950655
#             pickup_latitude: float,     # 40.783282
#             dropoff_longitude: float,   # -73.984365
#             dropoff_latitude: float,    # 40.769802
#             passenger_count: int):      # 1
#     """
#     we use type hinting to indicate the data types expected
#     for the parameters of the function
#     FastAPI uses this information in order to hand errors
#     to the developpers providing incompatible parameters
#     FastAPI also provides variables of the expected data type to use
#     without type hinting we need to manually convert
#     the parameters of the functions which are all received as strings
#     """
#     # $CHA_BEGIN

#     # ⚠️ if the timezone conversion was not handled here the user would be assumed to provide an UTC datetime
#     # create datetime object from user provided date
#     # pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

#     # localize the user provided datetime with the NYC timezone
#     eastern = pytz.timezone("US/Eastern")
#     localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

#     # convert the user datetime to UTC and format the datetime as expected by the pipeline
#     utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
#     formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

#     # fixing a value for the key, unused by the model
#     # in the future the key might be removed from the pipeline input
#     key = "2013-07-06 17:18:00.000000119"

#     X_pred = pd.DataFrame(dict(
#         key=[key],  # useless but the pipeline requires it
#         pickup_datetime=[formatted_pickup_datetime],
#         pickup_longitude=[pickup_longitude],
#         pickup_latitude=[pickup_latitude],
#         dropoff_longitude=[dropoff_longitude],
#         dropoff_latitude=[dropoff_latitude],
#         passenger_count=[passenger_count]))

#     model = app.state.model
#     X_processed = preprocess_features(X_pred)
#     y_pred = model.predict(X_processed)

#     # ⚠️ fastapi only accepts simple python data types as a return value
#     # among which dict, list, str, int, float, bool
#     # in order to be able to convert the api response to json
#     return dict(fare=float(y_pred))
#     # $CHA_END

#EC

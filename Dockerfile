
FROM tensorflow/tensorflow:2.10.0

COPY setup.py /setup.py
COPY chords_prog_proj /chords_prog_proj
COPY mlops /mlops
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -e .

CMD uvicorn chords_prog_proj.api.fast:app --host 0.0.0.0 --port $PORT

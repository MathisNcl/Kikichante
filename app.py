"""Front of the Web App"""

import random
import json

import pandas as pd
import streamlit as st
from transformers import Pipeline, pipeline

logger = st._logger.get_logger(__name__)
st._logger.set_log_level("INFO")

st.set_page_config(page_title="Bob or not bob?", page_icon="🎤")


with st.spinner("The model is being downloaded. Please wait."):

    @st.cache_resource
    def get_classifier() -> Pipeline:
        return pipeline("text-classification", model="MathNcl/Bob_or_not_Bob")

    classifier: Pipeline = get_classifier()

st.title("Bob or not Bob?")
st.header("Write some lyrics and find out if it could be a Bob Marley's song!")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def show_prediction(prediction: dict) -> None:
    asset = "assets/Not_bob.jpeg"
    if prediction["label"] == "Bob":
        if prediction["score"] > THRESHOLD:
            header = f"Bob - Score : {round(prediction['score'],2)}"
            asset = "assets/Bob.jpeg"
        else:
            header = f"Not Bob - Bob's score < {THRESHOLD} ({round(prediction['score'],2)})"
    else:
        header = f"Not Bob - Score : {round(prediction['score'],2)}"

    st.header(header)
    st.image(asset, use_column_width=True)


THRESHOLD: float = st.number_input(
    "Threshold for Bob", value=0.6, placeholder="Type a number...", min_value=0.5, max_value=1.0, step=0.01
)
text_input: str = st.text_input("Enter your lyrics👇")

if text_input:
    with st.spinner("Wait for it..."):
        prediction: dict = classifier(text_input)[0]
        logger.info(f"Input text : '{text_input}' ; Output : {json.dumps(prediction)}")
        show_prediction(prediction)

with st.expander("See methodology"):
    df_balance_label = pd.DataFrame(
        {"Bob": [423, 65, 113], "Not Bob": [528, 74, 169]}, index=["train", "test", "validation"]
    )

    df_training_infos = pd.read_csv("assets/result_train.txt")

    st.markdown(
        """
        Data are collected from [Genius](https://genius.com/) using [LyricsGenius](https://lyricsgenius.readthedocs.io/en/master/).

        To determine whether the text is from Bob Marley or not, lyrics from several singers have been collected.
        - 50 songs from Bob Marley
        - 10 songs for others:
            - Bruno Mars
            - Ed Sheeran
            - Elvis Presley
            - Michael Jackson
            - John Legend
            - Katy Perry
            - The Beatles

        Then, the text is cleaned and split into sentences for each artist.
        To add more context, some sentences are combined to create lyrics with a maximum of 100 characters.

        Labels are binary, indicating whether the lyrics are from Bob Marley or not.
        To avoid significant imbalance, only a maximum of 800 labels for each category are retained.

        The data is split into train, test, and validation sets.
        """
    )
    st.caption("Frequencies of every label for every dataset")
    st.dataframe(df_balance_label)
    st.markdown(
        """
        Model used is DistilBert (faster) with a binary head classification named [DistilBertForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertForSequenceClassification) train over 12 epochs.
        """
    )

    st.caption("Training informations")
    st.dataframe(df_training_infos)


def pick_random_lyrics(df: pd.DataFrame):
    index = random.randint(0, len(df) - 1)
    label, lyrics = df.iloc[index]
    st.session_state["label"] = label
    st.session_state["lyrics"] = lyrics


# Load the dataframe
df_game = pd.read_csv("data/lyrics.csv")


def create_game():
    if "label" not in st.session_state:
        pick_random_lyrics(df_game)

    st.write(f"'{st.session_state['lyrics']}'")
    col1, col2 = st.columns(2)

    if col1.button("Bob"):
        if st.session_state["label"] == "Bob":
            st.success("Correct!")
        else:
            st.error("Wrong Answer!")

    if col2.button("Not Bob"):
        if st.session_state["label"] != "Bob":
            st.success("Correct!")
        else:
            st.error("Wrong Answer!")


# PART 2 - GAMING ZONE
with st.container(border=True):
    st.header("Gaming zone: Is it Bob Marley's lyrics ?")

    if st.button("Give me new lyrics"):
        pick_random_lyrics(df_game)
    create_game()

import glob
import json
import os
import re

import pandas as pd


def get_cleaned_lyrics(lyrics: str, couplet=False) -> list[str]:
    """Clean the lyrics by removing some useless words and split into sentence.

    Args:
        lyrics (str): Raw lyrics from Genius Web site
        couplet (bool, optional): Whether to consider couplet. Defaults to False.

    Returns:
        list[str]: List of sentences of lyrics
    """
    cleaned_lyric: str = re.sub(r"\d{1,4} Contributors.{,100}Lyrics", "", lyrics)
    if not couplet:
        cleaned_lyric = re.sub(r"\[.{,50}\]", "", cleaned_lyric)
    cleaned_lyric = cleaned_lyric.replace("Embed", "").replace("You might also like", "")
    sentences: list[str] = cleaned_lyric.split("\n")
    if couplet:
        sentences = re.split(r"\[.{,50}\]", sentences)

    return [s for s in sentences if len(s) > 5]


def concat_lyrics_for_artist(df: pd.DataFrame, label: str, max_len: int = 512) -> list[str]:
    """Concatenate the lyrics from the dataframe to create sentences of max_len length by combining them in a row

    Args:
        df (pd.DataFrame): Initial dataframe with every sentence of the lyrics for every artist
        label (str): Label to concatenate lyrics (post processed artist column)
        max_len (int, optional): Max length of the sentence lyrics expected now. Defaults to 512.

    Returns:
        list[str]: List of lyrics concatenated
    """
    sub_df = df[df.labels == label]
    lyrics = []
    concatenated_lyrics = ""
    for _, row in sub_df.iterrows():
        if len(concatenated_lyrics) + len(row.lyrics) <= max_len:
            concatenated_lyrics += f"{row.lyrics} "
        else:
            lyrics.append(concatenated_lyrics)
            concatenated_lyrics = ""

    return lyrics


def create_final_df(path: str, max_len: int = 100) -> pd.DataFrame:
    """Create the final dataframe using utils functions to train the model with

    Args:
        path (str) : Folder path where are located lyrics json files
        max_len (int, optional): Max length of the sentence lyrics expected now. Defaults to 100.

    Returns:
        pd.DataFrame: Final Data frame
    """

    lyrics_df: dict = {"artist": [], "lyrics": [], "title": []}
    for path in glob.glob(os.path.join(path, "*.json")):
        with open(path) as f:
            artist_lyrics = json.load(f)

        artist = path.split("/")[-1].split(".")[0]

        for title, song in artist_lyrics.items():
            couplets = get_cleaned_lyrics(song)
            for c in couplets:
                lyrics_df["artist"].append(artist)
                lyrics_df["lyrics"].append(c)
                lyrics_df["title"].append(title)

    df = pd.DataFrame(lyrics_df)

    df["labels"] = df["artist"].apply(lambda x: "Bob" if x == "Bob_Marley_&_The_Wailers" else "Not_bob")

    new_df = pd.DataFrame()
    for label in df.labels.unique():
        df_artist = pd.DataFrame({"labels": label, "lyrics": concat_lyrics_for_artist(df, label, max_len=max_len)})
        new_df = pd.concat([new_df, df_artist])

    return new_df

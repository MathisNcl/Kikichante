import json

import lyricsgenius
from lyricsgenius.artist import Artist

token = "MyToken"
genius = lyricsgenius.Genius(token, retries=3)


def save_lyrics(artist: Artist):
    all_lyrics: dict[str, str] = {}
    name = artist.name.replace(" ", "_")
    for song in artist.songs:
        all_lyrics[song.title] = song.lyrics

    with open(f"data/{name}.json", "w") as f:
        json.dump(all_lyrics, f)
    print(f"{name}.json had been created.")


artists: list[str] = [
    "Ed Sheeran",
    "Bob Marley & The Wailers",
    "Elvis Presley",
    "John Legend",
    "Bruno Mars",
    "Michael Jackson",
    "The Beatles",
    "Katy Perry",
]

for artist in artists:
    print(artist)
    max_songs = 50 if artist == "Bob Marley & The Wailers" else 10
    retrieved_artist = genius.search_artist(artist, max_songs=max_songs, sort="popularity")
    save_lyrics(retrieved_artist)

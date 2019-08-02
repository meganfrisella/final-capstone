import dict_data as data
import pickle
from collections import Counter as c

# STILL MUST IMPORT DATA FROM DATABASE #

song_IDs = data.import_dictionaries('int_to_title')


def match(fingerprints, base):
    """
    Compares the recorded fingerprints to fingerprints from songs stored in the database to identify the song.

    Parameters:
    -----------
    fingerprints: List[Tuple[int, int, int]]
    List of fingerprints (freq_0, freq_1, dt) that store the fingerprints from the recorded audio file

    base: Dict[Tuple[int, int, int]:List[string, int]]
    Dictionary of fingerprints (freq_0, freq_1, dt) that map to a list of song names and time intervals.

    Returns:
    --------
    [string]: Title of the matched song in the form "(song title) by (artist)"
    """
    matches = []

    database = base

    for fp in fingerprints:
        if fp[:3] in database:
            for idx, val in enumerate(database[fp[0:3]]):
                matches.append((val[0], val[1]-fp[3]))

    max = c(matches).most_common()[0][1]
    if max > 20:
        return song_IDs[c(matches).most_common()[0][0][0]]
    else:
        return 'Sorry, we could not find your song :('

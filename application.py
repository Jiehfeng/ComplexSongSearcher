import sys
import threading
from multiprocessing.pool import ThreadPool

from tqdm import tqdm
import pandas as pd
import numpy as np

import json
import traceback

import psycopg2
import psycopg2.extras
from flask import Flask, jsonify, render_template, request, Response, send_file
import redis
import pickle

# Lyrical Similarity
from difflib import SequenceMatcher

# PEGASUS Transformer Model from Google Researchers (Liu & Zhao, 2020)
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

import warnings
warnings.filterwarnings("ignore")

model_name = "tuner007/pegasus_paraphrase"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)


# Database
conn = psycopg2.connect(database="song_database",
                        host="localhost",
                        user="jiehfeng",
                        password="jiehfeng@iit.ac.lk",
                        port="5432")

# Caching
# r = redis.Redis(host="localhost", port=6379, db=0, socket_timeout=100, socket_connect_timeout=100)


def paraphrase_lyrics(lyrics, number_of_variations):
    lyrical_phrases = tokenizer.prepare_seq2seq_batch([lyrics],
                                                      truncation=True,
                                                      padding="longest",
                                                      max_length=60,
                                                      return_tensors="pt").to(torch_device)
    num_beams = len(lyrics.split()) + 5
    processed_phrases = model.generate(**lyrical_phrases,
                                           max_length=100,
                                           num_beams=num_beams,
                                           num_return_sequences=number_of_variations,
                                           temperature=1.5)

    paraphrased_lyrics = tokenizer.batch_decode(processed_phrases, skip_special_tokens=True)
    return paraphrased_lyrics


# API
application = Flask(__name__)


@application.route("/")
def home_view():
    # Test Code
    '''print('[Song Searcher] - Loading dataset...')
    df = pd.read_csv("C:\\Users\\alain\\Desktop\Dataset Prepare\song_lyrics_processed.csv")
    print('[Song Searcher] - Dataset loaded.')
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    for index, song in tqdm(df.iterrows(), total=df.shape[0]):
        artist = str(song["artist"])
        title = str(song["title"])
        formatted_lyrics = json.dumps({"Lyrics": song["lyrics"]})

        if not artist or not title:
            continue
        else:
            print(artist + " - " + title)
            try:
                cur.execute("INSERT INTO songs (title, artist, lyrics) "
                            "VALUES (%s, %s, %s)", (title,
                                                    artist,
                                                    formatted_lyrics))
                conn.commit()
            except:
                print('[Song Searcher] - Sorry, song submission failed. Try again. (ERROR BELOW)')
                traceback.print_exc()
                cur.close()
                conn.rollback()

                should_continue = input("Continue? (Y/N)")
                if should_continue == "Y":
                    pass
                elif should_continue == "N":
                    break
                else:
                    print("ERROR.")
                    break'''



    return "<h1>Welcome to Song Searcher</h1>"


@application.route("/admin/submit-song", methods=["POST"])
def submit_song():
    admin_key = request.headers.get("ADMIN-KEY")

    if admin_key == 'jiehfeng@iit.ac.lk':
        title = request.args.get("title")
        artist = request.args.get("artist")
        lyrics = request.args.get("lyrics")

        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        '''cur.execute("CREATE TABLE songs ("
                       "song_no serial PRIMARY KEY,"
                       "title VARCHAR NOT NULL,"
                       "artist VARCHAR NOT NULL,"
                       "other JSON,"
                       "lyrics JSON NOT NULL"
                       ")")'''

        formatted_lyrics = json.dumps({"Lyrics": lyrics})

        paraphrased = ""
        sys.stderr.flush()
        print('[SONG SEARCHER] - Paraphrasing lyrics...')

        for line in tqdm(lyrics.splitlines()):
            if not line:
                continue
            parphrased_lyrics = paraphrase_lyrics(line, 4)
            for para in parphrased_lyrics:
                paraphrased += para + "\n"

        other = json.dumps({"Paraphrased Lyrics": paraphrased})

        print("ORIGINAL LYRICS: {}\n\n".format(lyrics))
        print("PARAPHRASED LYRICS: {}\n\n".format(paraphrased))

        try:
            cur.execute("INSERT INTO songs (title, artist, lyrics, other) VALUES (%s, %s, %s, %s)", (title,
                                                                                                     artist,
                                                                                                     formatted_lyrics,
                                                                                                     other))
            conn.commit()

            print("ORIGINAL LYRICS: {}\n\n".format(lyrics))
            print("PARAPHRASED LYRICS: {}\n\n".format(paraphrased))
        except:
            print('[Song Searcher] - Sorry, song submission failed. Try again. (ERROR BELOW)')
            traceback.print_exc()
            cur.close()
            conn.rollback()

    return "OK"


def process_paraphrase(lines):
    paraphrased = ""

    for line in tqdm(lines):
        if line:
            parphrased_lyrics = paraphrase_lyrics(line, 20)

            for para in parphrased_lyrics:
                paraphrased += para + "\n"

    return paraphrased


'''cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
cur.execute(
    "DELETE FROM songs"
)
conn.commit()'''


@application.route("/admin/get-song", methods=["POST"])
def get_song():
    admin_key = request.headers.get("ADMIN-KEY")

    if admin_key == 'jiehfeng@iit.ac.lk':
        title = request.args.get("title")
        artist = request.args.get("artist")
        lyrics = request.args.get("lyrics")

        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM songs")
        print(cur.fetchall())

        sent = {
            "Title": title,
            "Artist": artist,
        }

        if sent["Title"]:
            try:
                title = title.replace('"', '""').replace("'", "''")
                cur.execute("SELECT * FROM songs WHERE title = %s", (title,))
                song = cur.fetchall()

                return song[0]["lyrics"]["Lyrics"]
            except:
                print('[Song Searcher] - Sorry, song search failed. Try again. (ERROR BELOW)')
                traceback.print_exc()
                cur.close()
                conn.rollback()
        elif sent["Artist"]:
            try:
                print(artist)
                sql = "SELECT FROM songs WHERE artist = %s"
                cur.execute(sql, (artist,))
                song = cur.fetchall()

                return song[0]["lyrics"]["Lyrics"]
            except:
                print('[Song Searcher] - Sorry, song search failed. Try again. (ERROR BELOW)')
                traceback.print_exc()
                cur.close()
                conn.rollback()
        else:
            return "No data received."

    return "OK"


@application.route("/lyrics/search")
def search():
    query = request.args.get("SEARCH-QUERY")
    verbose = request.args.get("VERBOSE")
    mini_verbose = request.args.get("MINI-VERBOSE")

    print('[Song Searcher] - Fetching songs from the database...')
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cur.execute(
        "SELECT * FROM songs"
    )
    all_songs = cur.fetchall()
    print('[Song Searcher] - Songs successfully fetched.')
    print("\n\n---\n\n")

    # Paraphrase Query
    all_paraphrased = paraphrase_lyrics(query, 4)
    similarity_score_list = []
    direct_query_similarity_score_list = []

    for song_index, song in enumerate(tqdm(all_songs)):
        print('[Song Searcher] - SONG {}: {} - {}'.format(song_index, song['artist'], song['title']))
        phrases = str(song['lyrics']["Lyrics"]).splitlines()
        paraphrased_phrases = str(song['other']["Paraphrased Lyrics"]).splitlines()
        phrases = list(set(phrases))
        paraphrased_phrases = list(set(paraphrased_phrases))
        similarity_score_list.append(0)
        direct_query_similarity_score_list.append(0)

        for index, paraphrased in enumerate(all_paraphrased):
            for phrase in phrases:
                # Empty Lines
                if not phrase:
                    continue

                similarity = round(lyrical_similarity(phrase, paraphrased) * 100, 2)
                query_similarity = round(lyrical_similarity(phrase, query) * 100, 2)

                if similarity > 70:
                    similarity_score_list[song_index] += 1

                    if verbose == "True":
                        print('[Song Searcher] - ORIGINAL LYRICS: {} | PARAPHRASED LYRICS: {} | '
                              'QUERY: {}'.format(phrase,
                                                 paraphrased,
                                                 query))
                        print('[Song Searcher] - SIMILARITY: {} %'.format(similarity))
                        print('[Song Searcher] - For comparison...')
                        print('[Song Searcher] - QUERY AND ORIGINAL LYRIC SIMILARITY'
                              ' WITHOUT PARAPHRASING: {} %'.format(query_similarity))
                        print()
                elif query_similarity > 70:
                    direct_query_similarity_score_list[song_index] += 1
                    if verbose == "True":
                        print(
                            '[Song Searcher] - DIRECT QUERY SIMILARITY WAS OVER 50 % AT: {} %'.format(query_similarity))
                        print()

            for phrase in paraphrased_phrases:
                # Empty Lines
                if not phrase:
                    continue

                similarity = round(lyrical_similarity(phrase, paraphrased) * 100, 2)
                query_similarity = round(lyrical_similarity(phrase, query) * 100, 2)

                if similarity > 70:
                    similarity_score_list[song_index] += 1

                    if verbose == "True":
                        print('[Song Searcher] - ORIGINAL LYRICS: {} | PARAPHRASED LYRICS: {} | '
                              'QUERY: {}'.format(phrase,
                                                 paraphrased,
                                                 query))
                        print('[Song Searcher] - SIMILARITY: {} %'.format(similarity))
                        print('[Song Searcher] - For comparison...')
                        print('[Song Searcher] - QUERY AND ORIGINAL PARAPHRASED LYRIC SIMILARITY'
                              ' WITHOUT PARAPHRASING: {} %'.format(query_similarity))
                        print()

        if mini_verbose == "True":
            print('[Song Searcher] - SONG RESULTS FOR {}: {} - {}'.format(song_index, song['artist'], song['title']))
            print('[Song Searcher] - PARAPHRASING SCORE: {} | '
                  'DIRECT QUERY SCORE: {}'.format(similarity_score_list[song_index],
                                                  direct_query_similarity_score_list[song_index]))
            try:
                better_percentage = round(((similarity_score_list[song_index] - direct_query_similarity_score_list[
                    song_index]) / direct_query_similarity_score_list[song_index]) * 100, 2)
                print('[Song Searcher] - THIS PROGRAM PROVED TO BE {} % BETTER AT A '
                      'DIRECT SEARCH QUERY.'.format(better_percentage))
            except ZeroDivisionError:
                print('[Song Searcher] - THIS PROGRAM PROVED TO BE {} % BETTER AT A '
                      'DIRECT SEARCH QUERY.'.format(
                    (similarity_score_list[song_index] - direct_query_similarity_score_list[song_index]) * 100))
            print()

        print('\n---------\n')

    indices = len(all_songs)
    if indices > 5:
        indices = 5
    top_5_indices = np.argpartition(similarity_score_list, -indices)[-indices:]

    for song_index in top_5_indices:
        song = all_songs[song_index]

        print('[Song Searcher] - SONG {}: {} - {}'.format(song_index, song['artist'], song['title']))
        phrases = str(song['lyrics']["Lyrics"]).splitlines()
        paraphrased_phrases = str(song['other']["Paraphrased Lyrics"]).splitlines()
        phrases = list(set(phrases))
        similarity_score_list.append(0)
        direct_query_similarity_score_list.append(0)

        for index, paraphrased in enumerate(all_paraphrased):
            for phrase in phrases:
                # Empty Lines
                if not phrase:
                    continue

                similarity = round(lyrical_similarity(phrase, paraphrased) * 100, 2)
                query_similarity = round(lyrical_similarity(phrase, query) * 100, 2)

                if similarity > 50:
                    similarity_score_list[song_index] += 1
                elif query_similarity > 50:
                    direct_query_similarity_score_list[song_index] += 1

        print('[Song Searcher] - SONG RESULTS FOR {}: {} - {}'.format(song_index, song['artist'], song['title']))
        print('[Song Searcher] - PARAPHRASING SCORE: {} | '
              'DIRECT QUERY SCORE: {}'.format(similarity_score_list[song_index],
                                              direct_query_similarity_score_list[song_index]))
        try:
            better_percentage = round(((similarity_score_list[song_index] - direct_query_similarity_score_list[
                song_index]) / direct_query_similarity_score_list[song_index]) * 100, 2)
            print('[Song Searcher] - THIS PROGRAM PROVED TO BE {} % BETTER AT A '
                  'DIRECT SEARCH QUERY.'.format(better_percentage))
        except ZeroDivisionError:
            print('[Song Searcher] - THIS PROGRAM PROVED TO BE {} % BETTER AT A '
                  'DIRECT SEARCH QUERY.'.format(
                (similarity_score_list[song_index] - direct_query_similarity_score_list[song_index]) * 100))
        print()

    return "OK"


@application.route("/lyrics/test/paraphrase")
def paraphrase():
    global db
    lyrics = request.headers.get("Lyrics_Phrase")
    no_of_variations = int(request.headers.get("Variations"))

    paraphrased_lyrics = paraphrase_lyrics(lyrics, no_of_variations)

    formatted_string = ""
    for phrase in paraphrased_lyrics:
        formatted_string += phrase + "\n"

    return formatted_string


@application.route("/batch")
def batch():
    print("Reading CSV...")
    number_of_times = int(request.args.get("Number"))
    start = int(request.args.get("Start"))

    dataset = pd.read_csv("song_lyrics_processed.csv", skiprows=start, nrows=number_of_times)
    print("Read CSV.")

    batch_submit_func(dataset)


# Lyrical Sequence Rater
def lyrical_similarity(a, b):
    a_words = a.lower().split()
    b_words = b.lower().split()

    similar_count = 0
    total_count = len(a_words)    # Original

    for a_word in a_words:
        if a_word in b_words:
            similar_count += 1

    try:
        similarity_score = similar_count / total_count
    except ZeroDivisionError:
        similarity_score = 0

    return similarity_score
    # return SequenceMatcher(None, a, b).ratio()


# Batch Submission of Lyrics
def batch_submit_func(formatted_lyric_sets):
    title = request.args.get("title")
    artist = request.args.get("artist")
    lyrics = request.args.get("lyrics")

    for index, lyrics in enumerate(tqdm(formatted_lyric_sets, position=0)):
        formatted_lyrics = json.dumps({"Lyrics": lyrics["lyrics"]})

        paraphrased = ""
        sys.stderr.flush()
        print('[SONG SEARCHER] - Paraphrasing lyrics...')

        for line in tqdm(lyrics["lyrics"].splitlines(), position=1):
            if not line:
                continue
            parphrased_lyrics = paraphrase_lyrics(line, 4)
            for para in parphrased_lyrics:
                paraphrased += para + "\n"

        other = json.dumps({"Paraphrased Lyrics": paraphrased})

        print("ORIGINAL LYRICS: {}\n\n".format(lyrics["lyrics"]))
        print("PARAPHRASED LYRICS: {}\n\n".format(paraphrased))

        try:
            cur.execute("INSERT INTO songs (title, artist, lyrics, other) VALUES (%s, %s, %s, %s)", (lyrics["title"],
                                                                                                     lyrics["artist"],
                                                                                                     formatted_lyrics,
                                                                                                     other))
            conn.commit()

            print("ORIGINAL LYRICS: {}\n\n".format(lyrics["lyrics"]))
            print("PARAPHRASED LYRICS: {}\n\n".format(paraphrased))
        except:
            print('[Song Searcher] - Sorry, song submission failed. Try again. (ERROR BELOW)')
            traceback.print_exc()
            cur.close()
            conn.rollback()


# Lyrics Paraphraser
def lyrical_paraphraser(query):
    paraphrased_lyrics = paraphrase_lyrics(query, 20)

    return paraphrased_lyrics


if __name__ == "__main__":
    application.run(host="0.0.0.0", debug=False)

conn.close()

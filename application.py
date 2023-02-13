import json
import traceback

import psycopg2
import psycopg2.extras
from flask import Flask, jsonify, render_template, request, Response, send_file

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


def paraphrase_lyrics(lyrics, number_of_variations):
    lyrical_phrases = tokenizer.prepare_seq2seq_batch([lyrics],
                                                      truncation=True,
                                                      padding="longest",
                                                      max_length=60,
                                                      return_tensors="pt").to(torch_device)
    processed_phrases = model.generate(**lyrical_phrases,
                                       max_length=100,
                                       num_beams=20,
                                       num_return_sequences=number_of_variations,
                                       temperature=1.5)

    paraphrased_lyrics = tokenizer.batch_decode(processed_phrases, skip_special_tokens=True)
    return paraphrased_lyrics


# API
application = Flask(__name__)


@application.route("/")
def home_view():
    return "<h1>Welcome to Song Searcher</h1>"


@application.route("/admin/submit-song")
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

        try:
            cur.execute("INSERT INTO songs (title, artist, lyrics) VALUES (%s, %s, %s)", (title,
                                                                                          artist,
                                                                                          formatted_lyrics))
            conn.commit()
        except:
            print('[Song Searcher] - Sorry, song submission failed. Try again. (ERROR BELOW)')
            traceback.print_exc()
            cur.close()
            conn.rollback()

    return "OK"


@application.route("/admin/get-song")
def get_song():
    admin_key = request.headers.get("ADMIN-KEY")

    if admin_key == 'jiehfeng@iit.ac.lk':
        title = request.args.get("title")
        artist = request.args.get("artist")
        lyrics = request.args.get("lyrics")

        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        sent = {
            "Title": title,
            "Artist": artist,
        }

        if sent["Title"]:
            try:
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
                cur.execute("SELECT FROM songs WHERE artist = %s", (artist,))
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

    print('[Song Searcher] - Fetching songs from the database...')
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cur.execute(
        "SELECT * FROM songs"
    )
    all_songs = cur.fetchall()
    print('[Song Searcher] - Songs successfully fetched.')
    print("\n\n---\n\n")

    # Paraphrase Query
    all_paraphrased = lyrical_paraphraser(query)
    similarity_score_list = []
    direct_query_similarity_score_list = []

    for song_index, song in enumerate(all_songs):
        print('[Song Searcher] - SONG {}: {} - {}'.format(song_index, song['artist'], song['title']))
        phrases = str(song['lyrics']["Lyrics"]).splitlines()
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
                elif query_similarity > 50:
                    direct_query_similarity_score_list[song_index] += 1
                    if verbose == "True":
                        print('[Song Searcher] - DIRECT QUERY SIMILARITY WAS OVER 50 % AT: {} %'.format(query_similarity))
                        print()

        print('[Song Searcher] - SONG RESULTS FOR {}: {} - {}'.format(song_index, song['artist'], song['title']))
        print('[Song Searcher] - TOTAL POSSIBLE SCORE IS {}.'.format(len(phrases)))
        print('[Song Searcher] - PARAPHRASING SCORE: {} | '
              'DIRECT QUERY SCORE: {}'.format(similarity_score_list[song_index],
                                              direct_query_similarity_score_list[song_index]))
        try:
            better_percentage = round(((similarity_score_list[song_index] - direct_query_similarity_score_list[song_index]) / direct_query_similarity_score_list[song_index]) * 100, 2)
            print('[Song Searcher] - THIS PROGRAM PROVED TO BE {} % BETTER AT A '
                  'DIRECT SEARCH QUERY.'.format(better_percentage))
        except ZeroDivisionError:
            print('[Song Searcher] - THIS PROGRAM PROVED TO BE {} % BETTER AT A '
                  'DIRECT SEARCH QUERY.'.format((similarity_score_list[song_index] - direct_query_similarity_score_list[song_index]) * 100))
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


# Lyrical Sequence Rater
def lyrical_similarity(a, b):
    a_words = a.split()
    b_words = b.split()

    similar_count = 0
    total_count = len(a_words)    # Original

    for a_word in a_words:
        if a_word in b_words:
            similar_count += 1

    return similar_count / total_count
    # return SequenceMatcher(None, a, b).ratio()


# Lyrics Paraphraser
def lyrical_paraphraser(query):
    paraphrased_lyrics = paraphrase_lyrics(query, 20)

    return paraphrased_lyrics


if __name__ == "__main__":
    application.run(host="0.0.0.0", debug=False)

conn.close()

import json
from flask import Flask, jsonify, render_template, request, Response, send_file

# PEGASUS Transformer Model from Google Researchers (Liu & Zhao, 2020)
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = "tuner007/pegasus_paraphrase"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)


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


@application.route("/lyrics/paraphrase")
def paraphrase():
    lyrics = request.headers.get("Lyrics_Phrase")
    no_of_variations = int(request.headers.get("Variations"))

    paraphrased_lyrics = paraphrase_lyrics(lyrics, no_of_variations)

    formatted_string = ""
    for phrase in paraphrased_lyrics:
        formatted_string += phrase + "\n"

    return formatted_string


if __name__ == "__main__":
    application.run(host="0.0.0.0", debug=False)

import argparse
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import tweepy
from tensorflow import keras
from train_model import SEQ_LEN, clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TWEET_LEN = 240


class BotStreamer(tweepy.StreamListener):
    def on_status(self, status):
        logger.info(
            f"handling new event: from {status.user.screen_name}, {status.user.location}, {status.created_at}"
        )
        text = status.text.replace(app_handle, "")
        cleaned_text = clean_text(text)
        generated_text = generate_new_text(cleaned_text, diversity=0.45)
        reply = f"@{status.user.screen_name} {generated_text}"
        api.update_status(
            status=reply,
            in_reply_to_status_id=status.id,
            auto_populate_reply_metadata=True,
        )


def generate_new_text(seed, diversity):
    # pad or trim seed-sentence to 100 chars
    # since this is what the model was trained on
    if len(seed) > SEQ_LEN:
        generated = seed[-SEQ_LEN:]
    elif len(seed) < SEQ_LEN:
        generated = (SEQ_LEN - len(seed)) * " " + seed

    sequence = np.asarray([char_to_idx[char] for char in generated])

    for i in range(TWEET_LEN - len(seed)):
        preds = model.predict(sequence.reshape(-1, SEQ_LEN), verbose=0)[0][-1]
        next_char_index = sample(preds, diversity)
        next_char = idx_to_char[next_char_index]

        generated += next_char
        sequence = np.append(sequence[1:], next_char_index)

    return " ".join(generated.strip().split(" ")[:-1])[-TWEET_LEN:]


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    preds = np.exp(preds)
    preds = preds / np.sum(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    if not model_dir.exists:
        raise ValueError(
            "Please provide a valid input text file for argument 'model_dir'"
        )

    return model_dir


if __name__ == "__main__":
    model_dir = get_cli_args()

    # load secrets from system env
    consumer_key = os.environ.get("TWITTER_CONSUMER_KEY", "")
    consumer_secret = os.environ.get("TWITTER_CONSUMER_SECRET", "")
    access_token = os.environ.get("TWITTER_ACCESS_TOKEN", "")
    access_secret = os.environ.get("TWITTER_ACCESS_SECRET", "")
    app_handle = os.environ.get("TWITTER_APP_HANDLE", "@crnn_alice")

    # construct tweepy API instance
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)

    try:
        api.me()  # sanity check to make sure Twitter Oauth2 worked
    except tweepy.TweepError:
        raise ValueError(
            "Ooops, looks like there is an issue with your Twitter credentials.\n"
            "Better double check your environment variables"
        )

    # load model
    model = keras.models.load_model(model_dir)
    char_to_idx = joblib.load(model_dir / "char_to_idx")
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # listen for tweets to the app
    stream_listener = BotStreamer()
    stream = tweepy.Stream(auth, stream_listener)

    while True:
        stream.filter(track=[app_handle])
        time.sleep(5)

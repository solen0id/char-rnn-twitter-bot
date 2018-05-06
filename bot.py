import os
import keras
import tweepy
import joblib

import numpy as np

ALLOWED_CHARS = [' ', '!', '?', ',', '.', '\'', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


class BotStreamer(tweepy.StreamListener):

    def on_status(self, status):
        text = self.clean_text(status.text)
        generated = generate_new_text(text, 0.45)
        reply = "@{} {}".format(status.user.screen_name, generated)
        api.update_status(status=reply, in_reply_to_status_id=status.id, auto_populate_reply_metadata=True)

    @staticmethod
    def clean_text(text):
        text = text.replace("@need_me_some_hp", "").lower()
        return  "".join([char for char in text if char in ALLOWED_CHARS])

def generate_new_text(seed, diversity):

    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    # pad or trim seed-sentence to 100 chars
    # since this is what the model was trained on
    if len(seed) > 100:
        generated = seed[-100:]
    elif len(seed) < 100:
        generated = (100 - len(seed)) * " " + seed

    sequence = np.asarray([char_to_idx[char] for char in generated])

    for i in range(140-len(seed)):
        preds = model.predict(sequence.reshape(-1,100), verbose=0)[0][-1]
        next_char_index = sample(preds, diversity)
        next_char = idx_to_char[next_char_index]

        generated += next_char
        sequence = np.append(sequence[1:], next_char_index)

    return " ".join(generated.strip().split(" ")[:-1])[-140:]





# load secrets from system env
consumer_key = os.environ.get("TWITTER_CONSUMER_KEY")
consumer_secret = os.environ.get("TWITTER_CONSUMER_SECRET")
access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
access_secret = os.environ.get("TWITTER_ACCESS_SECRET")

# load model
model = keras.models.load_model("./models/jk_char_model2")
char_to_idx = joblib.load("./models/char_to_idx2")
idx_to_char = {idx:char for char, idx in char_to_idx.items()}

# construct api instance
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

# listen for tweets to the app
stream_listener = BotStreamer()
stream = tweepy.Stream(auth, stream_listener)
stream.filter(track=['@need_me_some_hp'])

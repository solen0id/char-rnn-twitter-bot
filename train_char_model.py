import re
import random
import joblib

import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Embedding, TimeDistributed
from keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import to_categorical
from keras import backend

from pathlib import Path

ALLOWED_CHARS = [' ', '!', '?', ',', '.', '\'', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def load_text(fpath):
    path = Path(fpath)
    text = path.read_text().lower()
    return text

def clean_text(text):
    return  "".join([char for char in text if char in ALLOWED_CHARS])

print("loading text...")
text = load_text("input.txt")
text = clean_text(text)

print("indexing characters...")
char_to_idx = {char:idx for idx, char in enumerate(sorted(set(text)))}   # string to set splits string into chars
idx_to_char = {idx:char for char, idx in char_to_idx.items()}
joblib.dump(char_to_idx, "models/char_to_idx_demo")

### build model ###
print("building model...")

SEQ_LEN = 100
STEP = 1
LAYER_COUNT = 3
HIDDEN_LAYER_SIZE = 256
DROPOUT = 0.2
VOCAB_SIZE = len(char_to_idx)

model = Sequential()
model.add(Embedding(VOCAB_SIZE, HIDDEN_LAYER_SIZE, input_shape=(SEQ_LEN,)))

for i in range(LAYER_COUNT):
    model.add(
        LSTM(
            HIDDEN_LAYER_SIZE,
            return_sequences=True,
            stateful=False,
        )
    )
    model.add(Dropout(DROPOUT))

model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer="adam")

### define model callbacks ###

def on_epoch_end(epoch, logs):

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    start_index = random.randint(0, len(text) - SEQ_LEN - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        sequence = np.asarray([char_to_idx[c] for c in text[start_index: start_index + SEQ_LEN]])
        seed = ''.join([idx_to_char[idx] for idx in sequence])

        print('----- Generating with seed:\n"{}"\n'.format(seed))
        print(seed, end='')

        for i in range(200):
            preds = model.predict(sequence.reshape(-1,SEQ_LEN), verbose=0)
            next_char_index = sample(preds[0][-1], diversity)
            next_char = idx_to_char[next_char_index]

            sequence = np.append(sequence[1:], next_char_index)
            print(next_char, end='')

        print('\n\n')

tensorboard_callback = TensorBoard(log_dir='./logs')
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
checkpoint_callback = ModelCheckpoint('./models/char_model_demo', monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks=[tensorboard_callback, print_callback, checkpoint_callback]

### generate trainingdata ###
print("generating char sequences for training...")

def make_sequences_w_targets(text, seq_len, step):
    X, y = [], []

    for i in range(0, len(text) - seq_len, step):
        char_seq = text[i:i+seq_len]
        next_char = text[i+seq_len]
        char_seq_shifted = char_seq[1:] + next_char
        char_seq_as_one_hot = [to_categorical(idx, num_classes=VOCAB_SIZE) for idx
                               in [char_to_idx[c] for c in char_seq_shifted]]

        X.append([char_to_idx[c] for c in char_seq])
        y.append(char_seq_as_one_hot)

    X = np.asarray(X).reshape(-1, seq_len)
    y = np.asarray(y)

    return X, y


X, y = make_sequences_w_targets(text, SEQ_LEN, STEP)

print("beginning training...")
model.fit(X, y, batch_size=64, epochs=10, callbacks=callbacks)

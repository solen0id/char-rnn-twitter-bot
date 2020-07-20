import argparse
import random
import re
from pathlib import Path
from typing import Tuple, Union

import joblib
import numpy as np
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    Dense,
    Dropout,
    Embedding,
    TimeDistributed,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

DROPOUT = 0.2
HIDDEN_LAYER_SIZE = 256
LAYER_COUNT = 3
SEQ_LEN = 100
STEP = 1
VOCAB_SIZE = 62  # determined by regex pattern used in clean_text()


def load_text(file_path: Union[str, Path]) -> str:
    return Path(file_path).read_text()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # replace all whitespace chars with space
    return re.sub(r"[^ a-zA-Z!?.,;:'\"-]", "", text)  # translates to 62 allowed chars


def on_epoch_end(epoch, logs):
    print(f"type epoch {type(epoch)}, type logs {type(logs)}")

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    # Function invoked at end of each epoch. Prints the generated text
    print()
    print("----- Generating text after Epoch: %d" % epoch)
    start_index = random.randint(0, len(text) - SEQ_LEN - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("----- diversity:", diversity)

        sequence = np.asarray(
            [char_to_idx[c] for c in text[start_index : start_index + SEQ_LEN]]
        )
        seed = "".join([idx_to_char[idx] for idx in sequence])

        print(f'----- Generating with seed:\n"{seed}"\n')
        print(seed, end="")

        for i in range(200):
            preds = model.predict(sequence.reshape(-1, SEQ_LEN), verbose=0)
            next_char_index = sample(preds[0][-1], diversity)
            next_char = idx_to_char[next_char_index]

            sequence = np.append(sequence[1:], next_char_index)
            print(next_char, end="")

        print("\n\n")


def make_sequences_w_targets(
    text: str, seq_len: int, step: int
) -> Tuple[np.array, np.array]:
    X, y = [], []

    for i in range(0, len(text) - seq_len, step):
        char_seq = text[i : i + seq_len]
        next_char = text[i + seq_len]
        char_seq_shifted = char_seq[1:] + next_char
        char_seq_as_one_hot = [
            to_categorical(idx, num_classes=VOCAB_SIZE)
            for idx in [char_to_idx[c] for c in char_seq_shifted]
        ]

        X.append([char_to_idx[c] for c in char_seq])
        y.append(char_seq_as_one_hot)

    X = np.asarray(X).reshape(-1, seq_len)
    y = np.asarray(y)

    return X, y


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--model-dir", default="./models/char-rnn-model/", type=str)
    parser.add_argument("--log-dir", default="./logs/", type=str)
    args = parser.parse_args()

    input_file = Path(args.input_file)
    model_dir = Path(args.model_dir)
    log_dir = Path(args.log_dir)

    if not input_file.exists:
        raise ValueError(
            "Please provide a valid input text file for argument 'input_file'"
        )

    for directory in [model_dir, log_dir]:
        # create model and log dirs if they don't exists already
        directory.mkdir(parents=True, exist_ok=True)

    return input_file, model_dir, log_dir


if __name__ == "__main__":
    print("main")
    input_file, model_dir, log_dir = get_cli_args()

    print("loading text...")
    text = load_text(input_file)
    text = clean_text(text)

    print("indexing characters...")
    char_to_idx = {char: idx for idx, char in enumerate(sorted(set(text)))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    joblib.dump(char_to_idx, model_dir / "char_to_idx")

    print("building model...")
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, HIDDEN_LAYER_SIZE, input_shape=(SEQ_LEN,)))

    for i in range(LAYER_COUNT):
        model.add(LSTM(HIDDEN_LAYER_SIZE, return_sequences=True, stateful=False,))
        model.add(Dropout(DROPOUT))

    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    ### define model callbacks ###
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    checkpoint_callback = ModelCheckpoint(
        model_dir, monitor="loss", verbose=1, save_best_only=True, mode="min",
    )
    callbacks = [tensorboard_callback, print_callback, checkpoint_callback]

    print("generating char sequences for training...")
    X, y = make_sequences_w_targets(text, SEQ_LEN, STEP)

    print("beginning training...")
    model.fit(X, y, batch_size=64, epochs=10, callbacks=callbacks)

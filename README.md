# Char-RNN Twitter Bot 🤖

This is a personal learning project to familiarize myself with RNNs operating on the
character level. The repository contains a script to train a LSTM Char-RNN on a public
domain text corpus (Alice in Wonderland), as well as a small script to serve the trained
model as a Twitter bot. The bot will listen to tweets directed at its own Twitter
handle and reply to any text sent its way with some generated text in the semantic
style of the training data.

## How to run the Project

### Prerequisites 🛠

In your virtual environment of choice, install all required dependencies with
`pip install -r requirements.txt`

If you want to serve the model via a Twitter Bot/App, you will need to register a
Twitter App before you get started with this project:
https://developer.Twitter.com/en/apps
In order to properly execute the Twitter Bot, you will need to set the following
environment variables, or store them in an env file (recommended), on the host where
you'd like to run the script or invoke the Docker image from:

- `Twitter_CONSUMER_KEY`: Twitter App Consumer API Key
- `Twitter_CONSUMER_SECRET`: Twitter App Consumer Secret
- `Twitter_ACCESS_TOKEN`: Twitter App Access Token, only visible when first created
- `Twitter_ACCESS_SECRET`: Twitter App Access Token Secret, only visible when first created
- `Twitter_APP_HANDLE`: The name of your Twitter App and the @handle used to listen at

If you want to leverage your GPU for model training, refer to the
[tensorflow-gpu documentation](https://www.tensorflow.org/install/gpu) for installation
references. Setting up CUDA and cuDNN exceeds the scope of this Readme.

#### Development

This project uses the opinionated [black](https://github.com/psf/black) formatter, as well
as some other pre-commit hooks that make life just a tad easier. To set both of these tools up
in one go, run `pip install pre-commit && pre-commit install`. Now all changes will be
formated on commit and you can focus on writing code.

### Training the model 📚

This step **does not require any Twitter App to be setup**. To train the model, make
sure you are in an activated virtualenv of your choosing with all prerequisites
installed. Invoke the training script with `python char-rnn-twitter-bot/train_model.py --input-file data/alice.txt` to train a small model on the public domain work "Alice's
Adventures in Wonderland" by Lewis Carroll. Of course you are free to train your model
on any other text corpus that is available to you. The `train_model.py` script takes
the following arguments:

- `--input-file` : The text file containing your training data text corpus, not pre-processed
- `--model-dir` : The directory where the model checkpoints and final model will be stored, defaults to `./models/char-rnn-model/`
- `--log-dir` : The directory where training logs will be saved ( You can use tensorboard to keep an eye on training)

Now you might be thinking: "But what about all the good stuff, like number of epochs, layer count, layer size, etc..."? Well, I'm personally not a huge fan of very long commands that span
a over more than a handful of arguments, so I'm deciding to keep those parameters as configuration in the code itself.
This has the added bonus of versioning changes in parameter values as well. If you'd rather configure these things when invoking the training script, feel free to add some arguments
to the argument parser as you see fit.

### Running the Twitter Bot 🐦

To run the Twitter Bot, you need two things: A registered Twitter App with credentials and a trained model. You can run the bot straight of your machine, or you can use the provided Docker image to make things a tad more portable.

To build the docker image, run the following command from the project root directory:
`docker build -t char-rnn-twitter-bot ./`

Once built, run the image with:
`docker run --env-file .env char-rnn-twitter-bot`
where `.env` is an environment file that contains the Twitter credentials. See `.env.example` for an example of the file structure that is expected.

#### Running Locally

Of course you can also run the bot locally without Docker with:
`python char-rnn-twitter-bot/bot.py models/char_model`
in an active virtualenv with all Twitter credentials exported as local environment
variables. Since the bot runs in a loop until manually interrupted, you might want to send
it to the background so you don't block your active shell. To run the bot in the background and even have it survive when the active shell is terminated, run:
`nohup python char-rnn-twitter-bot/bot.py models/char_model &`

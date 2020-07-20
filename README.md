# Char-RNN Twitter Bot 🤖

This is a personal learning project to familiarize myself with RNNs operating on the character level. The repository contains a script to train a LSTM Char-RNN on a public domain text corpus (Alice in Wonderland), as well as a small script to serve the trained model as a twitter bot. The bot will listen to tweets directed at its own twitter handle and reply to any text sent its way with some generated text in the semantic style of the training data.

## How to run the Project

### Prerequisites 🛠

In your virtual environment of choice, install all required dependencies with `pip install -r requirements.txt`

If you want to serve the model via a Twitter Bot/App, you will need to register a Twitter App before you get started with this project: https://developer.twitter.com/en/apps
In order to properly execute the Twitter Bot, you will need to set the following environment variables on the host where you'd like to run the script:

- `TWITTER_CONSUMER_KEY`: Twitter App Consumer API Key
- `TWITTER_CONSUMER_SECRET`: Twitter App Consumer Secret
- `TWITTER_ACCESS_TOKEN`: Twitter App Access Token, only visible when first created
- `TWITTER_ACCESS_SECRET`: Twitter App Access Token Secret, only visible when first created
- `TWITTER_APP_HANDLE`: The name of your Twitter App and the @handle used to listen at

If you want to leverage your GPU for model training, refer to the [tensorflow-gpu documentation](https://www.tensorflow.org/install/gpu) for installation references. Setting up CUDA and cuDNN exceeds the scope of this Readme.

### Training the model 📚

### Running the Twitter Bot 🐦

FROM python:3.8-slim

RUN useradd -m -d /home/twitter-bot twitter-bot
WORKDIR /home/twitter-bot/

COPY --chown=twitter-bot:twitter-bot requirements.txt .
RUN pip install --upgrade pip=='20.1.1' && pip install -r requirements.txt

COPY --chown=twitter-bot:twitter-bot models/ ./models/
COPY --chown=twitter-bot:twitter-bot char-rnn-twitter-bot/ .
COPY --chown=twitter-bot:twitter-bot entrypoint.sh .

USER twitter-bot

CMD ["./entrypoint.sh"]

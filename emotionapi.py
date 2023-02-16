# Bring in light weight dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import numpy as np
import pandas as pd
import re

app = FastAPI()

class ScoringItem(BaseModel):
    TweetText:str

# Loading the model
model = load_model('model.h5')

# Loading the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Function to remove special characters and links
def remove_special_characters(text):
    # Regular expression pattern for URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    # Replace URLs with an empty string
    text = re.sub(url_pattern, '', text)

    # Regular expression pattern for mentions starting with @
    mention_pattern = re.compile(r'@\w+')

    # Replace mentions with an empty string
    text = re.sub(mention_pattern, '', text)

    # Regular expression pattern for special characters
    special_char_pattern = re.compile(r'[^\w\s]+')

    # Replace special characters with an empty string
    text = re.sub(special_char_pattern, '', text)

    return text

@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    class_cat = {0 : 'Sadness', 1 : 'Joy', 2 : 'Love', 3 : 'Anger', 4 : 'Fear', 5 : 'Surprise'}
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    tweet_text = df['TweetText'][0]

    # Calling function to remove special characters
    tweet_text = remove_special_characters(tweet_text).lower()
    # Tokenizing the text
    data_seq = tokenizer.texts_to_sequences([tweet_text])
    data_pad = np.array(pad_sequences(data_seq, maxlen=66, padding='post'))
    prediction = model.predict(np.expand_dims(data_pad[0], axis=0))[0]
    pred_class = int(np.argmax(prediction).astype('uint8'))
    pred_category = class_cat[pred_class]

    return {"prediction_class" : int(pred_class),
            "prediction_category" : pred_category, 
            "text" : tweet_text}
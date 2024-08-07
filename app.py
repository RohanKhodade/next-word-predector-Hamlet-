import pickle
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


model=load_model(r"C:\Users\HP\OneDrive\Desktop\Artificial Intellegence\Deep learning\next word predector\hamlet_word_predector.h5")

# loading tokenizer

with open(r"C:\Users\HP\OneDrive\Desktop\Artificial Intellegence\Deep learning\next word predector\tokenizer.pkl","rb")as handle:
    tokenizer=pickle.load(handle)
    
    
# defining the function for predicting

max_len=14
def predict_next_word(model,tokenizer,text,max_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if (len(token_list)>=max_len):
        token_list=token_list[-(max_len-1):]
    token_list=pad_sequences([token_list],maxlen=max_len-1,padding="pre")
    
    predicted=model.predict(token_list)
    predicted_index=np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if (predicted_index==index):
            return word
    return None

# creating streamlit app

st.title("Predicting next word in Hamlet")
input_text=st.text_input("Enter the sequence of word","to be or not to be")
if st.button("Predict the word"):
    next_word=predict_next_word(model,tokenizer,input_text,max_len)
    st.write("Next word : {}".format(next_word))
    
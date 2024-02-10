import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer # use to lemmatize the words
from keras.models import load_model # Load_model is use to load the pre trained chatbot model

lemmatizer = WordNetLemmatizer()
intents = json.load(open('D:\AI_ML\Projects\ChatBot1\Project_1(ChatBot)\Include\intents.json'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbot_model.keras')

# clean up Function to tokenize the sentence and lemmatize for each word in tokenized sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bw = bag_of_words(sentence)
    res = model.predict(np.array([bw]))[0]

    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key = lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm Sorry, I don't Understand..."
    list_of_intents = intents_json['intents']
    tag = intents_list[0]['intent']
   
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result

print("Great! Bot is Running...")
print("Bot> Type 'Exit', to close the chat...")
flag = True
while(flag == True):
    message = input("You> ")
    if message == "exit":
        flag = False
    else:
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("Bot> ",res)
    



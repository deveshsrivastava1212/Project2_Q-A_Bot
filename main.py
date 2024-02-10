import random
import pickle
import json
import numpy as np, keras
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

data = json.loads(open('D:\AI_ML\Projects\ChatBot1\Project_1(ChatBot)\Include\intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in data['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
# Use Set Data Structure to remove all duplicates
words = sorted(set(classes))
classes = sorted(set(classes))
# Pickling the sorted words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb')) 

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words: bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(128, input_shape = (len(trainX[0]),),activation = 'relu' ))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation = 'softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, nesterov= True)

model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
hist = model.fit(np.array(trainX), np.array(trainY), epochs = 500, batch_size = 10, verbose =1)
model.save('chatbot_model.keras', hist)
print('Done')


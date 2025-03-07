import random
import json
import pickle

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers.legacy import SGD

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())  # load the json file into a json object or a dictionary in python
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']  # ignore these words

# loop through the intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # tokenize the words in the pattern
        words.extend(word_list)  # extend the word list with words from the pattern
        documents.append((word_list, intent['tag']))  # add the words and the tag to the documents list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # add the tag to the classes list

# Lemmatize and filter out ignore words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]

# Remove duplicates and sort the words
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))   
pickle.dump(classes,open('classes.pkl','wb'))

# nural network need numbers for a nural network to do that we use bag of words. we set teh word values to 0 or 1 if it occurs in the oattern 
training = []
output_empty = [0] * len(classes)
for document in documents :
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag,output_row])


random.shuffle(training)
training = np.array(training, dtype=object)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])

model =Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))  # relu is a rectified linear unit function this means that if the input is greater than 0 the output is the input
model.add(Dropout(0.5)) 
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(len(train_y[0]),activation='softmax')) # the function that helps us add up the results . its cales up the results to see how lokliy the label is found

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True) ## no idea why this is here
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy']) # we choose acccurecy as the metric to see how well the model is doing

model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1) # feed datat 200 times in a batch size of 5 and print the results
model.save('chatbot_model',model) # save the model
print('done')
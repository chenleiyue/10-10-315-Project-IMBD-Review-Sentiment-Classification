'''
mltrain_lstm_rnn.py

pls try replacing the imdb lib with the cleaned data from task and fill 
them in the testing log 3 if you would like to ?

'''

# rnn approach ( am trying different epoch sizes)
# reference : 
# https://github.com/geektown/keras-quick-startup


'''
Note : We are recording the training data from imdb lib, so 
       might overfit for the actual (testing) data
       
            ===== Testing log =====
            
TRIAL EPOCH   MAXLEN    FEATURE SPACE  BATCHSIZE     ACCURACY 
-------------------------------------------------------------
1 |   5       80        20000          32             84%
2 |   10      200       10000          32             81%
3 |  (delete this line; to be filled with task 1 testing)
 

'''



# Original lib :
'''
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(7)
'''

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np

# define feature space : 10000 （which is the size of our dictionary)

# define length of a feature sequence (which is the length of our vector)
maxlen = 200
# define batch size
batch_size = 32


# step 1 : use the data from imdb dataset for rnn walk through
#       -- note: this part will be replaced by cleaned data afterwards

# data reading initialization, use training file(not imdb) by default.
use_imdb = False
x_train = []
y_train = []
x_test = []
y_test = []
max_features = 0


print('Processing the cleaned data...')
if(use_imdb == True):
    (x_train,y_train),(x_test,y_test) = imdb.load_data(num_words= 10000)
else:
    #read docvec and s_tag_list
    x_train = np.load('docvec_list.npy', allow_pickle=True)
    y_train = np.load('s_tag_list.npy', allow_pickle=True)
    max_features = 73698


# The following print statement is to display what does the x, y vec looks like:
# print (x_train, y_train)


# printing some util information here: 
print(len(x_train),'training data set')
print(len(x_test),'testing data set')
print('Pad set ==> which is calculated by : (samples x time)')
x_train = sequence .pad_sequences(x_train ,maxlen= maxlen )
x_test = sequence .pad_sequences(x_test ,maxlen= maxlen )


'''
print('x_train shape:',x_train .shape )
print('x_test shape:',x_test .shape ) 
'''

print('Initializing modeling process')


model = Sequential()
# embeded layer makes the positive integer to be set vectors
model.add(Embedding (max_features ,128))
# set dropout
# I am not sure if it is necessary to try different values here
model.add(LSTM (128,dropout= 0.2,recurrent_dropout= 0.2))
# Dense layer
model.add(Dense(1,activation= 'sigmoid'))

model.compile(loss= 'binary_crossentropy',optimizer= 'adam',metrics= ['accuracy'])

print('Training...')
# fitting
# model.fit(x_train ,y_train ,batch_size= batch_size ,epochs= 10,validation_data= (x_test ,y_test ))
model.fit(x_train, y_train, batch_size=batch_size, epochs=10)

model.save("rnn_naive_docvec_model.h5")

score,acc = model.evaluate(x_train ,y_train ,batch_size= batch_size )
print('Score:',score)
print('Accuracy:', acc)

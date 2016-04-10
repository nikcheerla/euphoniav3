from keras.models import Graph
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
from keras.utils import np_utils
from keras.optimizers import RMSprop
import numpy as np
import random
import sys

import midi
import numpy as np
import random
import sys
import glob
from itertools import chain

import math

from keras import callbacks

remote = callbacks.RemoteMonitor(root='http://localhost:9000')

def round(x):
    return int(math.floor(x / 10.0)) * 10

def num_unique(arr):
	unique_ = []
	occurence_ = []
	num = 0

	for sub in arr:
	    if sub.tolist() not in unique_:
	        unique_.append(sub.tolist())
	        num = num + 1
	return num

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
	

#constants
maxlen = 20
note_range = 102


#gather musical notes

notes = []
for filename in glob.iglob("../music/*.mid"):
	print filename
	song = midi.read_midifile(filename)
	#print song
	song2 = chain.from_iterable(song)
	for event in song2:
		cur_type = type(event)
		if cur_type is midi.events.NoteOnEvent or cur_type is midi.events.NoteOffEvent:
			notes.append([event.tick, min(event.data[0], note_range - 1), event.data[1]])
	break
notes = np.array(notes)
#print notes

print "build semiredundant sequences of length maxlen"
data_len = len(notes) - maxlen

trainSeq = np.zeros((data_len, maxlen, 3), dtype=np.float32)
trainVal = np.zeros((data_len, 3), dtype=np.float32)

for i in range(0, data_len):
	trainSeq[i] = notes[i:i+maxlen]
	trainVal[i] = notes[i+maxlen]

trainTick = trainVal[:, 0]
trainNote = trainVal[:, 1]
trainVol = trainVal[:, 2]

trainTickSeq = trainSeq[:,:,0:1]
trainNoteSeq = trainSeq[:,:,1:2]
trainVolSeq = trainSeq[:,:,2:3]

#trainTick = np_utils.to_categorical(trainTick)
trainNote = np_utils.to_categorical(trainNote, nb_classes=note_range)
trainSwapped = np.swapaxes(trainNoteSeq, 0, 1)
print trainSwapped.shape

trainNoteSeq = np.zeros((maxlen, data_len, note_range))
for i in range(0, maxlen):
	trainNoteSeq[i] = np_utils.to_categorical(trainSwapped[i], nb_classes=note_range)

trainNoteSeq = np.swapaxes(trainNoteSeq, 0, 1)
print trainNoteSeq.shape

#trainVol = np_utils.to_categorical(trainVol)

trainTick = trainTick.astype(int)
trainNote = trainNote.astype(int)
trainVol = trainVol.astype(int)

#print sum(trainTick)
#print sum(trainNote)
#print sum(trainVol)

#print trainSeq.shape
#print "Length:", (trainNote).shape



# build the model: 2 stacked LSTM
print('Build model...')
model = Graph()
model.add_input(name = 'note_seq', input_shape=(maxlen,note_range))
model.add_input(name = 'vol_seq', input_shape=(maxlen,1))
model.add_input(name = 'tick_seq', input_shape=(maxlen,1))

model.add_node(LSTM(512, return_sequences=True, init='glorot_uniform'), name = 'lstm1', input = 'note_seq')
model.add_node(Dropout(0.2), name = 'drop1', input = 'lstm1')
model.add_node(LSTM(512, return_sequences=False, init='glorot_uniform'), name = 'lstm2', inputs=['drop1', 'tick_seq', 'vol_seq'], merge_mode = 'concat', concat_axis=2)
model.add_node(Dropout(0.2), name = 'drop2', input='lstm2')
model.add_node(Dense(300, activation = 'relu'), name = 'dense1', input = 'drop2')
model.add_node(Dense(1, activation='linear'), name = 'tick', input = 'dense1', create_output = True)
model.add_node(Dense(note_range, activation='softmax'), name = 'note', input = 'dense1', create_output = True)
model.add_node(Dense(1, activation='linear'), name = 'volume', input = 'dense1', create_output = True)


print('Compiling model')

rms = RMSprop(clipnorm = 2e5)
model.compile(optimizer=rms, loss={'note': 'categorical_crossentropy', 'tick': 'mse', 'volume': 'mse'})

print('Compiled model')

model.load_weights("cur_net_weights.h5")

length = 3000
start_index = 0

trainSeq2 = trainSeq[start_index:start_index + length]

trainNoteSeq2 = trainNoteSeq[start_index:start_index + length]
trainTickSeq2 = trainTickSeq[start_index:start_index + length]
trainVolSeq2 = trainVolSeq[start_index:start_index + length]

trainNote2 = trainNote[start_index:start_index + length]
trainTick2 = trainTick[start_index:start_index + length]
trainVol2 = trainVol[start_index:start_index + length]

print trainSeq2.shape
print trainNote2.shape
print trainTick2.shape
print trainVol2.shape
print trainNoteSeq2.shape
print trainTickSeq2.shape
print trainVolSeq2.shape

model.load_weights("cur_net_weights.h5")

model.fit({'note_seq': trainNoteSeq2, 'tick_seq': trainTickSeq2, 'vol_seq': trainVolSeq2,
	'note': trainNote2, 'tick': trainTick2, 'volume': trainVol2}, 
	batch_size=32, nb_epoch=8, callbacks=[remote])
model.save_weights("tuned_net_weights.h5", overwrite = True)


start_index = random.randint(0, data_len - 1)

sentence = trainSeq[start_index: start_index + 1]
for diversity in [0.2, 0.5, 0.75, 1.0, 1.2]:
	print('----- Generating from index: ' + str(start_index) + 'with diversity ' + str(diversity) + ' put in example.mid')

	pattern = midi.Pattern()
	track = midi.Track()
	pattern.append(track)

	for i in range(1, 500):
		testTickSeq = sentence[:,:,0:1]
		testNoteSeq = sentence[:,:,1:2]
		testVolSeq = sentence[:,:,2:3]

		testSwapped = np.swapaxes(testNoteSeq, 0, 1)

		testNoteSeq = np.zeros((maxlen, 1, note_range))
		for i in range(0, maxlen):
			testNoteSeq[i] = np_utils.to_categorical(testSwapped[i], nb_classes=note_range)

		testNoteSeq = np.swapaxes(testNoteSeq, 0, 1)

		predictions = model.predict({'note_seq': testNoteSeq, 'tick_seq': testTickSeq, 'vol_seq': testVolSeq});
		one_hot = predictions['note'][0]
		note_num = sample(one_hot, temperature = diversity)

		volume = int(predictions['volume']/5.0) * 5
		tick = int(predictions['tick'])

		note_num = min(note_num, 255, max(note_num, 1))
		volume = min(volume, 127, max(volume, 0))
		tick = min(tick, 255, max(tick, 0))

		note = [abs(tick), abs(note_num), abs(volume)]
		#print note
		sentence = np.append(sentence[:, 1:, ], [[note]], axis = 1)
		on = midi.NoteOnEvent(tick = note[0], pitch = note[1], velocity = note[2])
		track.append(on)

	eot = midi.EndOfTrackEvent(tick=1)
	track.append(eot)
	midi.write_midifile("tuned_example" + str(diversity) + ".mid", pattern)
		
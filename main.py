import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

word_to_emotions = {}
word_to_phonemes = {}
# emotion_to_class = {"anger":0, "fear":1, "anticipation":2, "trust":3, "surprise":4, "sadness":5, "joy":6, "disgust":7}
all_emotions = ["anger", "fear", "anticipation", "trust", "surprise", "sadness", "joy", "disgust"]
all_phonemes = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
NUM_PHONEMES = 69

# plan
# when loading dataset, do not load into one_hot
# just load into (word, emotion, phoneme) triplets
# then shuffle afterwards when necessary
# make into one_hot at the latest point

def emotions_to_vec(emotions):
    return [1 if e in emotions else 0 for e in all_emotions]
def phonemes_to_vec(phonemes):
    return [1 if p in phonemes else 0 for p in all_phonemes]

def load_dataset():
    with open('cmudict-0.7b', 'r', encoding="latin-1") as f:
        for line in f.readlines():
            word = str.split(line)[0]
            phonemes = str.split(line)[1:]
            word_to_phonemes[word] = phonemes
    
    with open('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r') as f:
        for line in f.readlines():
            word, emotion, assoc = re.split("\t", line)
            assoc = assoc[0] # removes backslash n
            if emotion != 'positive' and emotion != 'negative' and assoc[0] == '1' :
                print((word, emotion))
                word = word.upper()
                if word_to_phonemes.get(word, False):
                    entry = word_to_emotions.get(word, [])
                    if entry == []:
                        entry.append(emotion)
                    word_to_emotions[word] = entry

    valid_words = word_to_emotions.keys()
    #print([(word, ''.join(word_to_phonemes[word]), word_to_emotions[word][0]) for word in valid_words])

    X = [phonemes_to_vec(word_to_phonemes[word]) for word in valid_words]
    X = np.asarray(X, dtype=np.float32)

    Y = [emotions_to_vec(word_to_emotions[word]) for word in valid_words]
    Y = np.asarray(Y, dtype=np.float32)

    X, Y = shuffle(X, Y, random_state=0)

    X_train = X[:3500]
    X_test = X[3500:]
    Y_train = Y[:3500]
    Y_test = Y[3500:]
    # print(len(max(X, key=len)))
    # print(len(min(X, key=len)))

    return X_train, X_test, Y_train, Y_test

def main():
    X_train, X_test, Y_train, Y_test = load_dataset()

    model = keras.Sequential([
        keras.layers.Dense(12, input_dim=69, activation='relu'),
        keras.layers.Dense(8, activation='softmax')
        ])
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=150, batch_size=10)
    _, accuracy = model.evaluate(X_test, Y_test)
    print(accuracy)
    model.summary()

if __name__== "__main__":
    main()














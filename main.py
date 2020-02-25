import re
import numpy as np
import tensorflow as tf
from tensorflow import keras

word_to_emotions = {}
word_to_phonemes = {}
# emotion_to_class = {"anger":0, "fear":1, "anticipation":2, "trust":3, "surprise":4, "sadness":5, "joy":6, "disgust":7}
all_emotions = ["anger", "fear", "anticipation", "trust", "surprise", "sadness", "joy", "disgust"]
all_phonemes = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
NUM_PHONEMES = 69

def main():


    with open('cmudict-0.7b', 'r', encoding="latin-1") as f:
        for line in f.readlines():
            word = str.split(line)[0]
            phonemes = str.split(line)[1:]
            word_to_phonemes[word] = phonemes
    
    with open('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r') as f:
        for line in f.readlines():
            word, emotion, _ = re.split("\t", line)
            word = word.upper()
            if word_to_phonemes.get(word, False):
                entry = word_to_emotions.get(word, [])
                if entry == []:
                    entry.append(emotion)
                word_to_emotions[word] = entry

    phoneme_np = {}
    for word, phonemes in word_to_phonemes.items():
        phoneme_np[word] = [1 if p in phonemes else 0 for p in all_phonemes]

    X = [phoneme_np[word] for word in word_to_emotions.keys()]
    X = np.asarray(X, dtype=np.float32)
    print(X)
    print(X.shape)

    emotion_np = {}
    for word, emotions in word_to_emotions.items():
        emotion_np[word] = [1 if e in emotions else 0 for e in all_emotions]
    Y = [emotion_np[word] for word in word_to_emotions.keys()]
    Y = np.asarray(Y, dtype=np.float32)
    print(Y)
    print(Y.shape)

    model = keras.Sequential([
        keras.layers.Dense(12, input_dim=69, activation='relu'),
        keras.layers.Dense(8, activation='softmax')
        ])

    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    model.fit(X, Y, epochs=150, batch_size=10)
    _, accuracy = model.evaluate(X, Y)
    print(accuracy)
    model.summary()

if __name__== "__main__":
    main()














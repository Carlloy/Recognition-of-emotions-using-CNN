import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

emotions_labels = ['Złość', 'Zniesmaczenie', 'Strach', 'Radość', 'Smutek', 'Zaskoczenie', 'Obojętność']
emotion_data = []
fer_path = 'dataset/fer2013.csv'
data = pd.read_csv(fer_path)
emotion_data = data['emotion'].value_counts().sort_index()
y_pos = np.arange(len(emotions_labels))
plt.bar(y_pos, emotion_data, align='center', alpha=0.5)
plt.xticks(y_pos, emotions_labels, rotation='vertical')
plt.ylabel('Liczba')
plt.xlabel('Emocje')
plt.title('Liczba wzorców w zbiorze FER2013')
plt.show()


epoches = []
acc = []
loss = []
data = np.genfromtxt('results.csv', delimiter=',', names=['epoches', 'acc', 'loss'])
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(data['epoches']+1, data['acc'], color='r')
plt.xlabel('Iteracje')
plt.ylabel('Precyzja')
plt.title('Model - Precyzja')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(data['epoches']+1, data['loss'], color='r')
plt.xlabel('Iteracje')
plt.ylabel('Straty')
plt.title('Model - Straty')
plt.show()


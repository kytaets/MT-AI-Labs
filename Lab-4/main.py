import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, SimpleRNN, Reshape
matplotlib.use('TkAgg')

x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
z = np.cos(np.sin(y)) * np.sin(x)

def modelTesting(model):
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, z, epochs=20, batch_size=100)
    z_pred = model.predict(x)

    plt.plot(x, z, label='Actual')
    plt.plot(x, z_pred, label='Predicted')
    plt.legend()
    plt.title(model.name)
    plt.show()


def feedforwardCreation(layers, neurons):
    model = Sequential(name=f"Feedforward_{layers}x{neurons}")
    model.add(Input(shape=(1,)))
    for _ in range(layers):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, name='output'))
    return model


def cascadeforwardCreation(layers, neurons):
    inputLayer = Input(shape=(1,), name='input')
    current = Dense(neurons, activation='relu')(inputLayer)

    for _ in range(layers - 1):
        current = Concatenate()([inputLayer, current])
        current = Dense(neurons, activation='relu')(current)

    outputLayer = Dense(1, name='output')(current)
    model = Model(inputs=inputLayer, outputs=outputLayer, name=f"Cascade_{layers}x{neurons}")
    return model


def elmanCreation(layers, neurons):
    model = Sequential(name=f"Elman_{layers}x{neurons}")
    model.add(Input(shape=(1,)))
    model.add(Reshape((1, 1)))

    for _ in range(layers):
        model.add(SimpleRNN(neurons, return_sequences=True, activation='relu'))

    model.add(Dense(1, name='output'))
    model.add(Reshape((1,)))
    return model


if __name__ == "__main__":
    models = [
        feedforwardCreation(1, 10),
        feedforwardCreation(1, 20),
        cascadeforwardCreation(1, 20),
        cascadeforwardCreation(2, 10),
        elmanCreation(1, 15),
        elmanCreation(3, 5)
    ]

    for m in models:
        print(f"\n🔹 Навчання моделі: {m.name}")
        modelTesting(m)

from tensorflow.keras.layers import LSTM, Dense, ConvLSTM2D, Bidirectional, Input
from tensorflow.keras.models import Sequential
import tensorflow_docs as tfdocs
import matplotlib.pyplot as plt
import tensorflow_docs.plots
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np

class ModelsHanlder:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def save_model_and_score(self, model, X_test, y_test, path='./models/result.json'):
        loss, mse, mae = model.evaluate(X_test, y_test, verbose=0)


        if not os.path.exists(path): 
          raise ValueError(f'Path problem with {path}')
        with open(path, 'r') as f: scores = json.load(f)
        scores[self.name] = {"mse": mse, "mae":mae}
        with open(path, 'w') as f: json.dump(scores, f)
        model.save(f'models/{self.name}')

    def check_right_direction(self, model, X_test, y_test):
      ""

    def plot_history_model(self, history_model):
         plot_obj = tfdocs.plots.HistoryPlotter(smoothing_std=2)
         fig, axes = plt.subplots(1,1,figsize=(15,5))
         plot_obj.plot({f'{self.name} ': history_model}, metric = "mean_squared_error")

    @staticmethod
    def load_model(name):
      return tf.keras.models.load_model(f'models/{name}')

    def vanilla_LSTM(self, input_shape):
        self.name = "Vanilla LSTM"
        model = Sequential([
                    Input(shape=input_shape, name='lstm_input'),
                    LSTM(units=50, name='Vanilla LSTM',
                        activation='tanh', recurrent_activation = "sigmoid",  
                        return_sequences=False, batch_size=self.batch_size),
                    Dense(units=1),
                    ])
        
        return model
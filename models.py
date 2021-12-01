from tensorflow.keras.layers import LSTM, Dense, ConvLSTM2D, Bidirectional
from keras.models import Sequential
import tensorflow_docs as tfdocs
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np

from tensorflow.keras.layers import LSTM, Dense, ConvLSTM2D, Bidirectional
from keras.models import Sequential
import tensorflow_docs as tfdocs
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np

class ModelsHanlder:
    def __init__(self, optimizer, loss, metrics, batch_size):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics,
        self.batch_size = batch_size

    def create_bidirectional(self, input_shape):
        self.name = "Bidirectional LSTM"
        model = Sequential([
                    Input(shape=input_shape, name='lstm_input'),
                    Bidirectional(LSTM(30, 
                                       name='Bidirectional_LSTM',
                                       activation='tanh', 
                                        return_sequences=True, 
                                        recurrent_activation = "sigmoid",  
																				batch_size=self.batch_size)),
                    Dense(1)
                    ])
        model.compile(optimizer = self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def save_model_and_score(self, model, x, y, epoch: int = 50, patience:int = 50):
          try: tf.keras.utils.plot_model(model, show_shapes=True)
          except Exception as e: print("Not able to plot the model.")
          history_model = model.fit(x=x, y=y, epochs=epoch, validation_split = 0.1, verbose=1)
          return history_model


    def plot_history_model(self, history_model):
         plot_obj = tfdocs.plots.HistoryPlotter(smoothing_std=2)
         fig, axes = plt.subplots(1,1,figsize=(15,5))
         plot_obj.plot({f'{self.name} ': history_model}, metric = "mean_squared_error")
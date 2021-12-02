from tensorflow.keras.layers import LSTM, Dense, ConvLSTM2D, Bidirectional, Input
from tensorflow.keras.models import Sequential
import tensorflow_docs as tfdocs
import matplotlib.pyplot as plt
import tensorflow_docs.plots
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import json
import os

class ModelsHanlder:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def save_model_and_score(self, model, X_test, y_test, target_scalar, ticket, path='./models/'):
        loss, mse, mae = model.evaluate(X_test, y_test, verbose=0)
        direction_score = self.check_direction(model, X_test, y_test, target_scalar)
        path_folder = f"{path}/{ticket}"
        path_result = f"{path_folder}/result.json"
        path_model = f"{path_folder}/{self.name}.json"
        if not os.path.exists(path_folder): os.mkdir(path_folder)
        if not os.path.exists(path_result): 
          with open(path_result, 'w') as f: 
            json.dump({self.name:{"mse":mse, "mae":mae, "scores":direction_score}}, f)
        else:
          with open(path_result, 'r') as f: scores = json.load(f)
          with open(path_result, 'w') as f: 
            scores[self.name] = {"mse": mse, "mae":mae, "scores":direction_score}
            json.dump(scores, f)
        model.save(path_model)
        return direction_score

    def check_direction(self, model, X_test, y_test, target_scalar):
      res, pred = dict(), model.predict(X_test)
      for itruth in range(1, len(X_test)-1):
        truth_today = target_scalar.inverse_transform(X_test[itruth][-1][1].reshape(-1, 1))
        truth_tomorrow = target_scalar.inverse_transform(X_test[itruth+1][-1][1].reshape(-1, 1))
        prediction = target_scalar.inverse_transform(pred[itruth-1].reshape(-1, 1))
        real_flow = truth_tomorrow-truth_today
        predicted_flow = truth_tomorrow-prediction
        res[itruth] = {"correct": 0}
        res[itruth]["real-predicted"] = (real_flow-predicted_flow)[0][0]
        if (real_flow < 0 and predicted_flow < 0) or (real_flow > 0 and predicted_flow > 0):
          res[itruth]["correct"] = 1
      results = pd.DataFrame(res).T
      ret = results['real-predicted'].describe().to_dict()
      ret['mean_correct'] = results.correct.mean()
      return ret

    def plot_history_model(self, history_model):
         plot_obj = tfdocs.plots.HistoryPlotter(smoothing_std=2)
         fig, axes = plt.subplots(1,1,figsize=(15,5))
         plot_obj.plot({f'{self.name} ': history_model}, metric = "mean_squared_error")

    @staticmethod
    def plot_test_pred(model, X, y):
      fig,axes= plt.subplots(1,1,figsize=(20,15))
      prediction = model.predict(X)
      sns.lineplot(y=prediction.flat,x=[i for i in range(len(prediction))], ax=axes)
      sns.lineplot(y=y.flat,x=[i for i in range(len(prediction))], ax=axes, color='red', linewidth =0.3)

    @staticmethod
    def load_model(ticket, name):
      return tf.keras.models.load_model(f'models/{ticket}/{name}')

    def vanilla_LSTM(self, input_shape):
        self.name = "Vanilla_LSTM"
        model = Sequential([
                    Input(shape=input_shape, name='lstm_input'),
                    LSTM(units=50, name='Vanilla_LSTM',
                        activation='tanh', recurrent_activation = "sigmoid",  
                        return_sequences=False, batch_size=self.batch_size),
                    Dense(units=1),
                    ])
        
        return model
import tensorflow_docs as tfdocs
import matplotlib.pyplot as plt
import tensorflow_docs.plots
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os


class ModelsHandler:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def save_model_and_score(self, model, X_test, y_test, target_scalar, ticket, path='./models/'):
        loss, mse, mae = model.evaluate(X_test, y_test, verbose=0)
        direction_score, score_df = self.check_direction(model, X_test, y_test, target_scalar)
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
        return direction_score, score_df

    def check_direction(self, model, X_test, y_test, target_scalar):
      res = dict()
      for itruth in range(len(X_test)-1):
        truth_today = target_scalar.inverse_transform(y_test[itruth].reshape(-1, 1))
        truth_tomorrow = target_scalar.inverse_transform(y_test[itruth+1].reshape(-1, 1))
        prediction = target_scalar.inverse_transform(preds[itruth].reshape(-1, 1))
        real_sign = "up" if (truth_today-truth_tomorrow) < 0 else 'down'
        predicted_sign = "up" if (prediction-truth_tomorrow) < 0 else 'down'
        real_flow = (truth_today-truth_tomorrow)[0][0]
        predicted_flow = (prediction-truth_tomorrow)[0][0]
        res[itruth] = dict(correct=real_sign==predicted_sign, 
                          real_flow=real_flow, 
                          predicted_flow=predicted_flow,
                          mse=(real_flow - predicted_flow)**2)
      res = pd.DataFrame(res).T
      return {'correct': res.correct.mean(), 'scores': res.mse.astype('float64').describe().to_dict()}, res

    def plot_history_model(self, history_model):
         plot_obj = tfdocs.plots.HistoryPlotter(smoothing_std=2)
         fig, axes = plt.subplots(1,1,figsize=(15,5))
         plot_obj.plot({f'{self.name} ': history_model}, metric = "mean_squared_error")

    def draw_model(self, model, ticket):
      path_folder = f"models/{ticket}"
      if not os.path.exists(path_folder): os.mkdir(path_folder)
      return tf.keras.utils.plot_model(model, show_shapes=True, to_file=f"models/{ticket}/{self.name}.png")

    def load_model(self, ticket, name):
      self.name = name
      return tf.keras.models.load_model(f'models/{ticket}/{name}.json')

    @staticmethod
    def plot_test_pred(model, X, y, target_scalar):
      fig,axes= plt.subplots(1,1,figsize=(20,15))
      prediction = model.predict(X)
      inverse_scalar_pred = target_scalar.inverse_transform(prediction.reshape(-1,1))
      inverse_target = target_scalar.inverse_transform(y.reshape(-1,1))
      days = [i for i in range(len(prediction))]
      sns.lineplot(y=inverse_scalar_pred.flat, x=days, ax=axes, label = 'predicted', linewidth=1.5)
      sns.lineplot(y=inverse_target.flat, x=days, ax=axes, color='red', linewidth=1.2, label = 'real')
      for d in days: plt.axvline(d, 0, alpha=0.3, color='black')
      plt.xlim(-1, len(days))
      plt.ylim(np.min(inverse_scalar_pred)-30, np.max(inverse_target)+30)
      plt.legend()
      plt.show()
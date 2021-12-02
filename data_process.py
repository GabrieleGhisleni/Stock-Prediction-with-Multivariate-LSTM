from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class DataPreProcess:
    def __init__(self, scaler = StandardScaler, timesteps: int = 20):
        self.scaler = scaler()
        self.scaler_target = scaler()
        self.scaler_trained = False
        self.timesteps = timesteps
    
    def scale(self, data, target, timestamp, scaler = StandardScaler) -> pd.DataFrame:
        self.scaler_trained = True
        res =  pd.DataFrame(self.scaler.fit_transform(data), columns = data.columns)
        res.loc[:, ['Close']] = self.scaler_target.fit_transform(target.values.reshape(-1,1))
        res['Date'] = timestamp
        return res

    def create_windows(self, data, col_x, col_y):
      x,y = [],[] 
      windows = (len(data)-self.timesteps)
      for iwindow in range(windows):
        x.append(data.loc[:, col_x].values[iwindow: iwindow+self.timesteps])
        y.append(data.loc[:, col_y].values[iwindow+self.timesteps])
      x, y = np.array(x), np.array(y)
      assert (x.shape[0] == y.shape[0])
      print(f"Shape of transormed data: x={x.shape}, y={y.shape}")
      return (x,y)

    @staticmethod
    def test(df, N=40):
      return df.iloc[0:-N, :], df.iloc[-N:, :]
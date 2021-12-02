from tensorflow.keras.layers import LSTM, Dense, ConvLSTM2D, Bidirectional, Input, Flatten
from tensorflow.keras.models import Sequential


class ModelsDispatch(ModelsHandler):
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def vanilla_LSTM(self, input_shape):
        self.name = "Vanilla_LSTM_50"
        return  Sequential([
                    Input(shape=input_shape, name='lstm_input'),
                    LSTM(units=50, name='Vanilla_LSTM', batch_size=self.batch_size),
                    Dense(units=25, activation='relu'),
                    Dense(units=1),
                    ])
        
    def stacked_LSTM(self, input_shape):
        self.name = "stacked_LSTM"
        return Sequential([
                    Input(shape=input_shape, name='lstm_input'),
                    LSTM(units=50, name='stacked_LSTM', return_sequences=True, batch_size=self.batch_size),
                    LSTM(units=30, name='stacked_LSTM_2',batch_size=self.batch_size),
                    Dense(units=1),
                    ])
        
    def bidirectional_LSTM(self, input_shape):
        self.name = "bidirectional_LSTM"
        return Sequential([
                    Input(shape=input_shape, name='lstm_input'),
                    Bidirectional(LSTM(units=50, name='bidirectional_LSTM', batch_size=self.batch_size)),
                    Dense(units=20),
                    Dense(units=1),
                    ])
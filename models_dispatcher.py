from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input, Flatten
from tensorflow.keras.models import Sequential
from models_wrap import ModelsHandler

class ModelsDispatch(ModelsHandler):
    """
    Models dispatch, see the evaluation on the test to understand which models
    performs better.
    """
    def __init__(self, batch_size):
        super().__init__(batch_size)


    def light_dense_vanilla_LSTM(self, input_shape, name= "light_dense_vanilla_LSTM"):
        self.name = name
        return  Sequential([
                    Input(shape=input_shape, name='lstm_input'),
                    LSTM(units=200, name='Vanilla_LSTM', batch_size=self.batch_size),
                    Dense(units=10, activation='leaky_relu'),
                    Dense(units=1),
                    ])

    def light_dense_vanilla_LSTM_softmax(self, input_shape, name= "light_dense_vanilla_LSTM_softmax"):
        self.name = name
        return  Sequential([
                    Input(shape=input_shape, name='lstm_input'),
                    LSTM(units=200, name='Vanilla_LSTM', batch_size=self.batch_size),
                    Dense(units=20, activation='softmax'),
                    Dense(units=10, activation='softmax'),
                    Dense(units=1),
                    ])
        

    def stacked_LSTM(self, input_shape, name="stacked_LSTM"):
        self.name = name
        return Sequential([
                    Input(shape=input_shape, name='lstm_input'),
                    LSTM(units=50, name='stacked_LSTM', return_sequences=True, batch_size=self.batch_size),
                    LSTM(units=30, name='stacked_LSTM_2',batch_size=self.batch_size),
                    Dense(units=1),
                    ])
        

    def bidirectional_LSTM(self, input_shape, name="bidirectional_LSTM"):
        self.name = name
        return Sequential([
                    Input(shape=input_shape, name='lstm_input'),
                    Bidirectional(LSTM(units=50, name='bidirectional_LSTM', batch_size=self.batch_size)),
                    Dense(units=20),
                    Dense(units=1),
                    ])
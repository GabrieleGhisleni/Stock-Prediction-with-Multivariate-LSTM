import yfinance as yf
import seaborn as sns
import pandas as pd
import numpy as np


class FinanceFetch:
    def __init__(self, ticket: str):
        self.ticket_name = ticket
        self.yahoo_ticket = yf.Ticker(ticket)

    def fetch(self, period: str, plot: bool=True, year: str="2021") -> pd.DataFrame:
        self.history = self.yahoo_ticket.history(period)
        self.history.drop(columns=['Dividends', 'Stock Splits', 'High', 'Low'], inplace=True)
        self.history.reset_index(inplace=True)
        if plot: self.plot_flow(year=year)
        return self.history

    def plot_flow(self, year: str = "2021") -> None:
        fig,axes = plt.subplots(1,1, figsize=(20,10))
        sns.lineplot(data=self.history.loc[self.history.Date > "2021"], y='Open', x='Date', ax = axes, color='firebrick')
        sns.lineplot(data=self.history.loc[self.history.Date > "2021"] , y='Close', x='Date', ax = axes)
        plt.title(f"Flow of {self.ticket_name} in {year}")
        plt.show()


 


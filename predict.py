import warnings
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

warnings.simplefilter(action="ignore", category=FutureWarning)


class predict(object):
    def __init__(self, dataframe):
        self.prophet_df = dataframe.reset_index().rename(
            columns={"Date": "ds", "Adj Close": "y"}
        )
        self.model = Prophet()

    def prediction(self, period):
        self.model.fit(self.prophet_df)
        future = self.model.make_future_dataframe(periods=period)
        forecast = self.model.predict(future)
        fig = plot_plotly(self.model, forecast)
        return fig
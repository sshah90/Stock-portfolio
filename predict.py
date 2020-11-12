import warnings
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.plot import add_changepoints_to_plot


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
        # Eliminate weekend from future dataframe
        future['day'] = future['ds'].dt.weekday
        future = future[future['day']<=4]
        forecast = self.model.predict(future)
        fig = plot_plotly(self.model, forecast)
        fig2 = self.model.plot_components(forecast)
        fig3 = self.model.plot(forecast)
        a = add_changepoints_to_plot(fig3.gca(), self.model, forecast)
        return (fig, fig2,fig3)

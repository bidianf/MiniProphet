#MiniProphet

[MiniProphet](https://github.com/bidianf/miniprophet.git) is a minimalistic version of [Facebook's Prophet](https://facebook.github.io/prophet/) time series forecasting model. Prophet is a Bayesian model with (possible)  non-linear logistic trends that capture well the patterns of product adoption and diffusion. It enables automatic trend break detection and rich seasonality patterns, making it a popular forecasting tool. MiniProphet is written entirely in Python, removing the dependency on the probabilistic modeling language Stan and the associated C++ libraries that can introduce complexity and fragility. Therefore, MiniProphet is suitable for production environments where additional bells and whistles such as plotting capabilities are not needed.

MiniProphet calculates the Maximum A Posteriori (MAP) parameter estimates via L-BFGS-B (from scipy.optimize) without calling Stan. It does not support a full MCMC sampling of the posterior distribution. Confidence intervals around forecasts are obtained by simulating the trend and idiosyncratic uncertainty. MiniProphet modifies only two modules, "forecaster.py" and "models.py". As with Prophet, it accepts user specified holidays replacing the built-in ones. If access to the Prophet built-in holidays is desired, one can add the relevant modules from Prophet with minor modifications.

The documentation of Prophet carries over to MiniProphet (except any references to MCMC sampling). The source code repository for Prophet is located at https://github.com/facebook/prophet, from where built-in holidays, diagnostics tools and plotting tools can be added to MiniProphet if desired.

MiniProphet can be installed directly from the repo  using

`pip install git+https://github.com/bidianf/miniprophet.git#egg=miniprophet`

Example usage

  >>> import pandas as pd
  >>> from forecaster import Prophet 
  >>> df = pd.read_csv('data.csv') # df is a pandas.DataFrame with 'y' and 'ds' columns
  >>> from forecaster import Prophet 
  >>> m = Prophet()
  >>> m.fit(df)  
  >>> future = m.make_future_dataframe(periods=365)
  >>> df2 = m.predict(future)
  >>> df2.head()# miniprophet

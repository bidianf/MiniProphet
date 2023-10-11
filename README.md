# MiniProphet

[MiniProphet](https://github.com/bidianf/miniprophet.git) is a minimalistic version of [Facebook's Prophet](https://facebook.github.io/prophet) time series forecasting tool that is written entirely in Python and removes the dependence on the probabilistic modeling language Stan and the associated C++ libraries, making it suitable for production environments. It allows for non-linear logistic trends that capture well the patterns of product adoption and diffusion, user specified holidays, can detect trend breaks automatically and captures seasonality patterns at multiple frequencies. 

Miniprophet calculates the Maximum A Posteriori (MAP) parameter estimates via L-BFGS-B (from scipy.optimize) and does not support a full MCMC sampling of the posterior distribution. Confidence intervals around forecasts are obtained by simulating the trend and idiosyncratic uncertainty. It modifies only two Prophet modules, "forecaster.py" and "models.py".  If access to the Prophet built-in holidays is desired, one can add the relevant modules from Prophet with minor modifications. You can install miniprophet from the repo directly using

```
pip install git+https://github.com/bidianf/miniprophet.git#egg=miniprophet
```

Example usage:
```
  >>> import pandas as pd
  >>> from forecaster import Prophet 
  >>> df = pd.read_csv('data.csv') # df is a pandas.DataFrame with 'y' and 'ds' columns
  >>> from forecaster import Prophet 
  >>> m = Prophet()
  >>> m.fit(df)  
  >>> future = m.make_future_dataframe(periods=365)
  >>> df2 = m.predict(future)
  >>> df2.head()
```

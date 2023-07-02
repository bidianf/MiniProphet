# miniProphet
MiniProphet (https://github.com/bidianf/miniprophet.git)is a minimalistic version of Prophet, suitable for production environments where bells and whistles such as plotting capabilities are not needed, and where dependence on the probabilistic modeling language Stan and the associated C++ libraries introduce complexity and fragility. Prophet,  released by Facebook's Core Data Science team  as  open source software (https://code.facebook.com/projects/),  is   a Bayesian model for time series forecasting allowing for non-linear logistic trends that capture well the patterns of product adoption and diffusion. Therefore, it suits the forecasting needs of many companies. It also allows for rich seasonality patterns which are prominent in practice.  

Miniprophet is written entirely in Python and removes the dependencies on Stan. Miniprophet obtains the Maximum A Posteriori (MAP) parameter estimates via L-BFGS-B (from scipy.optimize) without calling Stan. It does not support a full MCMC sampling of the posterior distribution. Therefore, confidence intervals around forecasts are obtained by simulating the trend and idiosyncratic uncertainty, but not seasonality uncertainty which is supported only by MCMC sampling.

MiniProphet modifies only two modules, "forecaster.py" and "models.py". As with Prophet, it accepts user specified holidays replacing the built-in ones. If access to the Prophet built-in holidays is desired, one can add the relevant modules from Prophet with minor modifications.

The documentation of Prophet carries over to miniprophet (except any references to MCMC sampling), and is available at  https://facebook.github.io/prophet/. The source code repository for Prophet is located at   https://github.com/facebook/prophet, from where built-in holidays, diagnostics tools and plotting tools can be added if desired.

# Example usage

```python
  >>> import pandas as pd
  >>> from forecaster import Prophet 
  >>> df = pd.read_csv('data.csv') # df is a pandas.DataFrame with 'y' and 'ds' columns
  >>> from forecaster import Prophet 
  >>> m = Prophet()
  >>> m.fit(df)  
  >>> future = m.make_future_dataframe(periods=365)
  >>> df2 = m.predict(future)
  >>> df2.head()


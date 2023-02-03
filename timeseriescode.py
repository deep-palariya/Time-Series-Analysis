import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the time series data
data = pd.read_csv('time_series_data.csv')

# Convert the data into a time series object
ts = data['Value']

# Fit the ARIMA model
model = ARIMA(ts, order=(1,1,1))
model_fit = model.fit()

# Print summary of the model
print(model_fit.summary())

# Plot the residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

# Plot the residuals density
residuals.plot(kind='kde')
plt.show()

# Perform a normality test on the residuals
from scipy.stats import normaltest
stat, p = normaltest(residuals)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# Make predictions
forecast = model_fit.forecast(steps=10)[0]

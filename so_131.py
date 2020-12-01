import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
import math
df = pd.read_csv ('teamdata.csv')

y = df['R']
x = df['SO']

x_1=sm.add_constant(x)
x_1.head()

model = sm.OLS(y,x_1)
results = model.fit()
results
print(results.params)

y_hat=results.predict(x_1)
plt.scatter(x,y)
plt.title('Strike Outs vs Runs')
plt.xlabel("OBP")
plt.ylabel("Runs")
plt.plot(x,y_hat, "r")
plt.show()





import numpy as np
import pandas as pd

t = np.linspace(0, 6 * np.pi, 100)
input_1 = np.sin(t)
input_2 = np.sin(t + np.pi/3)
output_1 = np.cos(t)
output_2 = np.cos(t + np.pi/3)

# Set up empty DataFrame
data = pd.DataFrame({'Column_1' : []})

data['Column_' + str(0)] = input_1
data['Column_' + str(1)] = input_2
data['Column_' + str(2)] = output_1
data['Column_' + str(3)] = output_2

data.to_csv('myData.csv', index=False, header=False)
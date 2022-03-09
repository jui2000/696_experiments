import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_normal = pd.read_csv("data_normal.csv")

X = np.array(data_normal[["x_1", "x_2"]])
Y = np.array(data_normal["y"])

data_0 = X[Y==0,:]
data_1 = X[Y==1,:]

label_0 = Y[Y==0]
label_1 = Y[Y==1]

plt.scatter(data_0[:,0], data_0[:,1])
plt.scatter(data_1[:,0], data_1[:,1])
plt.show()

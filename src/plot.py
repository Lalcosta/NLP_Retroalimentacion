import matplotlib.pyplot as plt
import pandas as pd

loss=pd.read_csv("errors.csv")
plt.plot(loss["Epoch"],loss["Train_loss"], label="Train loss")
plt.plot(loss["Epoch"],loss["Ev_loss"], label="Train errors")
plt.title("Test error vs train error")
plt.show()
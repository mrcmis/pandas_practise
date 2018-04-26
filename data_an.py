import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

train_data_file = "crx.data.csv"
training_df = pd.read_csv(train_data_file,
                          names=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13',
                                 'A14', 'A15', 'A16'])
print(training_df.head(100))
print(training_df.tail(100))

training_df.info()
print(training_df.describe())

print(training_df.A4.nunique())
print(training_df.A4.unique())


def print_basic_histogram(data):
    plt.figure(figsize=(8, 4))
    plt.hist_params = {'normed': False, 'bins': 20, 'alpha': 0.4}
    plt.hist(data.dropna())
    plt.ylabel('count ')
    plt.xlabel(' occurrance ')
    plt.show()

def pairplot(data, vars):
  plot_kws={'alpha': 0.5,
            'marker': '.'}
  sns.pairplot(data.dropna(),
               vars=vars,
               diag_kind="kde",
               size=3, plot_kws=plot_kws)
  plt.show()



print_basic_histogram(training_df.A3)
pairplot(training_df, ["A3","A8"])


print(training_df.groupby('A16').size())

print(training_df.loc[training_df['A16'] == '+']['A3'].mean())
print(training_df.loc[training_df['A16'] == '-']['A3'].mean())

training_df["A17"] = training_df.A3 / training_df.A15

training_df["A18"] = np.where(np.logical_and(training_df.A16 == "+", training_df.A1 == "a"),
                              "A",
                              "B")

print(training_df.head(1000))

print(training_df.head(100))

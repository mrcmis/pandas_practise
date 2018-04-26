import matplotlib.pyplot as plt
import pandas as pd

train_data_file = "crx.data.csv"
training_df = pd.read_csv(train_data_file,
                          names=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13',
                                 'A14', 'A15', 'A16'])
print(training_df.head(100))
print(training_df.tail(100))

training_df.info()

print(training_df.A4.nunique())
print(training_df.A4.unique())


def print_basic_histogram(data):
    plt.figure(figsize=(10, 6))
    plt.hist_params = {'normed': False, 'bins': 20, 'alpha': 0.4}
    plt.hist(data.dropna())
    plt.ylabel('count ')
    plt.xlabel(' occurrance ')
    plt.show()


print_basic_histogram(training_df.A3)

print(training_df.groupby('A16').size())

print(training_df.loc[training_df['A16'] == '+']['A3'].mean())
print(training_df.loc[training_df['A16'] == '-']['A3'].mean())

training_df["A17"] = training_df.A3/ training_df.A15
print(training_df.head(100))

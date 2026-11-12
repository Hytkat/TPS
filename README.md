# TPS

# Program 1: Load and Explore a Dataset
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("First 5 rows:")
print(df.head())

print("Shape (rows, columns):", df.shape)

print("Information about the data:")
df.info()

print("Summary statistics:")
print(df.describe())

# Program 2: Data Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

plt.hist(df['sepal length (cm)'])
plt.title("Histogram of Sepal Length")
plt.show()

plt.scatter(df['petal length (cm)'], df['petal width (cm)'])
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

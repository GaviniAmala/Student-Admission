from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

# load the dataset
data = pd.read_csv('binary.csv')

# extract features and target
X = data[['gre', 'gpa', 'rank']].values
y = data['admit'].values

# create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# plot the data points
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

# set labels and title
ax.set_xlabel('GRE Score')
ax.set_ylabel('GPA')
ax.set_zlabel('Rank')
ax.set_title('Admission Dataset')

# show the plot
plt.show()
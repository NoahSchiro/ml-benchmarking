import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Needed for test train split
from sklearn.neighbors import KNeighborsClassifier   # Needed for the kNN algo
from sklearn.metrics import accuracy_score			 # Needed to show the accuracy of the model

# Always good to set a seed for reproducibility
SEED = 7
np.random.seed(SEED)

# Loading Data
df = pd.read_csv('./diabetes.csv')

# Getting dataframe columns names
df_name=df.columns

X = df[df_name[0:8]]
Y = df[df_name[8]]

# Split into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,
                                                    test_size=0.25,
                                                    random_state=0,
                                                    stratify=df['Outcome'])

for i in range(1, 10):
	model = KNeighborsClassifier(n_neighbors=i)

	model.fit(X_train, Y_train)

	Y_prediction = model.predict(X_test)

	print("{} has accuracy: {}".format(i, accuracy_score(Y_test, Y_prediction)))

# # Create model with k=3
# model = KNeighborsClassifier(n_neighbors=7)

# # Train the model using the training sets
# model.fit(X_train, Y_train)

# # Predict
# Y_prediction = model.predict(X_test)

# # Print accuracy
# print(accuracy_score(Y_test, Y_prediction))
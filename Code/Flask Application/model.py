import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Read the data 
clinical_data = pd.read_csv('heart.csv')
X = clinical_data.iloc[:,:13].values
y = clinical_data["target"].values

# Dividing into train-test splits
train_x,test_x,train_y, test_y = train_test_split(X,y,test_size = 0.3 , random_state = 0 )

# Scale model to standarize before giving to model
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

print("Finished scaling. Going to start building model")

# Building model
model = Sequential()
model.add(Dense(activation = "relu", input_dim = 13, 
                     units = 8, kernel_initializer = "uniform"))
model.add(Dense(activation = "relu", units = 14, 
                     kernel_initializer = "uniform"))
model.add(Dense(activation = "sigmoid", units = 1, 
                     kernel_initializer = "uniform"))
model.compile(optimizer = 'adam' , loss = 'binary_crossentropy', 
                   metrics = ['accuracy'] )

model.summary()
print("Going to start training")
model.fit(train_x , train_y , batch_size = 8 , epochs = 100  )


# save model
save_path = './model.h5'
model.save(save_path)


predicted_y = model.predict(test_x)
predicted_y = (predicted_y > 0.5)


conf_matrix = confusion_matrix(test_y, predicted_y)
conf_matrix

accuracy = (conf_matrix[0][0]+conf_matrix[1][1])/(conf_matrix[0][1] + conf_matrix[1][0] +conf_matrix[0][0] +conf_matrix[1][1])
print(accuracy*100)
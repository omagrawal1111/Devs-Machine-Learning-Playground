#%%
import numpy as np 
import pandas as pd 
# %%
import os 
for dirname, _,filenames in os.walk(r'C:\Users\omagr\OneDrive\Documents\GitHub\Devs-Machine-Learning-Playground\Sign Language Recognition\Dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# %%
import pandas as pd 
train_csv = pd.read_csv(r'C:\Users\omagr\OneDrive\Documents\GitHub\Devs-Machine-Learning-Playground\Sign Language Recognition\Dataset\sign_mnist_train\sign_mnist_train.csv')
test_csv = pd.read_csv(r'C:\Users\omagr\OneDrive\Documents\GitHub\Devs-Machine-Learning-Playground\Sign Language Recognition\Dataset\sign_mnist_test\sign_mnist_test.csv')
# %%
labels = []
images =[]
for row in train_csv.iterrows():
    label = row[1][0]
    image = np.array_split(row[1][1:],28)
    labels.append(label)
    images.append(image)

#Get the number of unique classes
num_classes = len(np.unique(labels))

labels = np.array(labels)
images = np.array(images)
print(labels.shape)
print(images.shape)

# %%
#Expand dims for these 2 np arrays so that they can be set as input to TF model
labels = np.expand_dims(labels,axis=1)
images = np.expand_dims(images,axis=3)
print(labels.shape)
print(images.shape)


# %%
#Create X_test (images) and Y_test (labels); we will also use this for Validation

labels_test = []
images_test = []
for row in test_csv.iterrows():
    label = row[1][0]
    image = np.array_split(row[1][1:],28)
    labels_test.append(label)
    images_test.append(image)

labels_test = np.array(labels_test)
images_test = np.array(images_test)
print(labels_test.shape)
print(images_test.shape)


# %%
#Expand dims for these 2 np arrays so that they can be set as input to TF model
labels_test = np.expand_dims(labels_test,axis=1)
images_test = np.expand_dims(images_test,axis=3)
print(labels_test.shape)
print(images_test.shape)


# %%
#Convert from str to FLoat
X_train = images.astype(float)
Y_train = labels.astype(float)
X_test = images_test.astype(float)
Y_test = labels_test.astype(float)
# %%
from sklearn.model_selection import train_test_split
X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 12345)
# %%
print(X_train.shape)
print(X_validate.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_validate.shape)
print(Y_test.shape)
# %%

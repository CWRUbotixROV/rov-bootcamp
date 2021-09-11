import pickle

from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import transformers
import utils

# create an instance of each transformer
hogify = transformers.HogTransformer(
    pixels_per_cell=(10, 10),
    cells_per_block=(2, 2),
    orientations=9,
    block_norm='L2-Hys'
)
scalify = StandardScaler()

paths, labels = utils.get_all_training_data()
images = []
for path in paths:
    images.append(imread(path))

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)

print(len(images))

# call fit_transform on each transform converting X_train step by step
X_train_hog = hogify.fit_transform(X_train)
X_train_prepared = scalify.fit_transform(X_train_hog)

X_test_hog = hogify.transform(X_test)
X_test_prepared = scalify.transform(X_test_hog)

with open("scaler.dat", 'wb') as scaler_file:
    pickle.dump(scalify, scaler_file)

with open("training_data.dat", 'wb') as training_file:
    pickle.dump((X_train_prepared, y_train), training_file)

with open("testing_data.dat", 'wb') as testing_file:
    pickle.dump((X_test_prepared, y_test), testing_file)

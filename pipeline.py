import os

import joblib
import numpy as np
from skimage.io import imread
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from transformers import RGB2GrayTransformer, HogTransformer


def load_data(src):
    training_labels = []
    training_images = []
    for subdir in os.listdir(src):
        for file in os.listdir(f"{src}/{subdir}"):
            im = imread(f"{src}/{subdir}/{file}")
            training_labels.append(subdir)
            training_images.append(im)
    return np.array(training_images), np.array(training_labels)


param_grid = [
    {
        'hogify__orientations': [8, 9],
        'hogify__cells_per_block': [(2, 2), (3, 3)],
        'hogify__pixels_per_cell': [(8, 8), (10, 10), (12, 12)],
        # 'classify': [
        #             SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
        #             svm.SVC(kernel='linear')
        # ]
    }
]

HOG_pipeline = Pipeline([
    ('grayify', RGB2GrayTransformer()),
    ('hogify', HogTransformer(
        pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm='L2-Hys')
     ),
    ('scalify', StandardScaler()),
    ('classify', SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
])

X_train, y_train = load_data("processed")
X_test, y_test = load_data("images/evaluation")
print("Images loaded")

grid_search = GridSearchCV(HOG_pipeline,
                           param_grid,
                           cv=3,
                           n_jobs=1,
                           scoring='accuracy',
                           verbose=3,
                           return_train_score=True)

grid_res = grid_search.fit(X_train, y_train)

print("Saving file")

joblib.dump(grid_res, 'hog_sgd_model.pkl')

print("Stats:")
print(grid_res.best_estimator_)
print(grid_res.best_score_)
print(grid_res.best_params_)

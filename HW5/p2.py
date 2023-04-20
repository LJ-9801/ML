import cv2
import numpy as np
from numpy import unique
import os
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import KFold

def img_preprocess(img_path):
    img = cv2.imread(img_path)
    # squash image to [1,0]
    img = np.array(img)/255
    # flatten image
    img = img.reshape(-1,3)

    return img

def get_best_model(components, img):
    best_model = None
    best_score = -np.inf
    kf = KFold(n_splits=10)
    kf.get_n_splits(img.T)
    for component in components:
        print(f"Number of components: {component}")
        gmm = GMM(n_components=component)
        score = []
        for train_idx, test_idx in kf.split(img):
            x_train, x_test = img[train_idx], img[test_idx]
            gmm.fit(x_train)
            score.append(gmm.score(x_test))
        mean_score = np.mean(score)
        print(f"avg score of {component} components 10-Fold Cross Validation: {mean_score}")
        if mean_score > best_score:
            best_score = mean_score
            best_model = gmm

    return best_model, best_score


if __name__ == "__main__":

    components = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    img_path = os.getcwd() + "/image/train.jpg"
    og_img = cv2.imread(img_path)
    
    img = img_preprocess(img_path)
    best_model, best_score = get_best_model(components, img)

    labels = best_model.predict(img)
    n = best_model.n_components
    labels = labels.reshape(og_img.shape[0], og_img.shape[1])*255/n

    labels = cv2.GaussianBlur(labels, (5,5), 0)
    edges = cv2.Canny(np.uint8(labels), 100, 200)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new = np.zeros((og_img.shape[0], og_img.shape[1], 3))+255

    for i in range(len(contours)):
        cv2.drawContours(og_img, contours, i, (0,255,0), 3)

    for i in range(len(contours)):
        cv2.drawContours(new, contours, i, (0,0,0), 3)

    cv2.imwrite("contours.jpg", new)
    cv2.imwrite("segmented.jpg", og_img)


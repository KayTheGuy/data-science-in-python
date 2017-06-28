import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer


OUTPUT_TEMPLATE = (
    'Bayesian classifier: {bayes_rgb:.3g} {bayes_lab:.3g}\n'
    'kNN classifier:      {knn_rgb:.3g} {knn_lab:.3g}\n'
    'SVM classifier:      {svm_rgb:.3g} {svm_lab:.3g}\n'
)


# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 114, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[
                           np.uint8, np.uint8, np.uint8])


def plot_predictions(model, lum=71, resolution=256):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; 
    plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of LAB colour values, with L=lum
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    # convert to RGB for consistency with original input
    X_grid = lab2rgb(lab_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((wid * hei, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at L=%g' % (lum,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.ylabel('B')
    plt.imshow(X_grid.reshape((hei, wid, 3)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)


def get_data():
    if len(sys.argv) >= 2:
        data = pd.read_csv(sys.argv[1])
        X = data[['R', 'G', 'B']].values / 255
        y = data['Label'].values
        return X, y
    else:
        print('Unable to read the csv file: provide a valid csv file name in the same directory!')
        sys.exit()


def RGB_to_LAB(rgb_values):
    # reshape as a 2D image with rgb values (3D in total)
    rgb_values.shape = (rgb_values.shape[0], 1, 3)
    lab_values = rgb2lab(rgb_values)
    # reshape the lab values to 1 tuple of rgb values (2D in total)
    lab_values.shape = (lab_values.shape[0], 3)
    # reshape back the original data to 1 tuple of rgb values (2D in total)
    rgb_values.shape = (rgb_values.shape[0], 3)
    return lab_values


def get_bayes_rgb_model():
    return GaussianNB()


def get_bayes_lab_model():
    # make pipeline for LAB color space
    model_lab = make_pipeline(
        FunctionTransformer(RGB_to_LAB),
        GaussianNB()
    )
    return model_lab


def get_knn_rgb_model():
    # based on experiments 8 gave the best score for k between 2 and 10
    return KNeighborsClassifier(n_neighbors=8)


def get_knn_lab_model():
    # make pipeline for LAB color space
    model_lab = make_pipeline(
        FunctionTransformer(RGB_to_LAB),
        # based on experiments 8 gave the best score for k between 2 and 10
        KNeighborsClassifier(n_neighbors=8)
    )
    return model_lab


def get_svc_rgb_model():
    # based on experiments 10^3 gave the best score for C between 10^(-3) and
    # 10^3
    return SVC(kernel='linear', decision_function_shape='ovr', C=1e3)


def get_svc_lab_model():
    # make pipeline for LAB color space
    model_lab = make_pipeline(
        FunctionTransformer(RGB_to_LAB),
        # based on experiments 10 gave the best score for C between 10^(-3) and
        # 10^3
        KNeighborsClassifier(n_neighbors=10)
    )
    return model_lab


def main():
    X, y = get_data()

    # split data into training and test data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    bayes_rgb_model = get_bayes_rgb_model()
    bayes_lab_model = get_bayes_lab_model()
    knn_rgb_model = get_knn_rgb_model()
    knn_lab_model = get_knn_lab_model()
    svc_rgb_model = get_svc_rgb_model()
    svc_lab_model = get_svc_lab_model()

    # train each model and output image of predictions
    models = [bayes_rgb_model, bayes_lab_model, knn_rgb_model,
              knn_lab_model, svc_rgb_model, svc_lab_model]
    for i, m in enumerate(models):
        m.fit(X_train, y_train)
        plot_predictions(m)
        plt.savefig('predictions-%i.png' % (i,))

    print(OUTPUT_TEMPLATE.format(
        bayes_rgb=bayes_rgb_model.score(X_test, y_test),
        bayes_lab=bayes_lab_model.score(X_test, y_test),
        knn_rgb=knn_rgb_model.score(X_test, y_test),
        knn_lab=knn_lab_model.score(X_test, y_test),
        svm_rgb=svc_rgb_model.score(X_test, y_test),
        svm_lab=svc_lab_model.score(X_test, y_test),
    ))


if __name__ == '__main__':
    main()

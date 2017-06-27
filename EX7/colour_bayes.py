import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer


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
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution

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
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.imshow(X_grid.reshape((hei, wid, 3)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.imshow(pixels)


def RGB_to_LAB(rgb_values):
    # reshape as a 2D image with rgb values (3D in total)
    rgb_values.shape = (rgb_values.shape[0], 1, 3)
    lab_values = rgb2lab(rgb_values)
    # reshape back to 1 tuple of rgb values (2D in total)
    lab_values.shape = (lab_values.shape[0], 3)
    return lab_values


def main():

    if len(sys.argv) >= 2:
        data = pd.read_csv(sys.argv[1])
    else:
        print('Unable to read the csv file: provide a valid csv file name in the same directory!')
        return -1

    # reshape R G B and normalize to range [0-1]
    R = (data['R'] / 255)[:, np.newaxis]
    G = (data['G'] / 255)[:, np.newaxis]
    B = (data['B'] / 255)[:, np.newaxis]

    # array with shape (n, 3). Divide by 255
    X = np.concatenate((R, G, B), axis=1)
    y = data['Label']  # array with shape (n,) of colour words

    # split data into training and test data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # create Naive Bayes classifier
    model_rgb = GaussianNB()
    model_rgb.fit(X_train, y_train)

    # make pipeline for LAB color space
    model_lab = make_pipeline(
        FunctionTransformer(RGB_to_LAB),
        GaussianNB()
    )
    model_lab.fit(X_train, y_train)

    # print accuracy score for model_rgb and model_lab
    print(model_rgb.score(X_test, y_test))
    print(model_lab.score(X_test, y_test))

    plot_predictions(model_rgb)
    plt.savefig('predictions_rgb.png')
    plot_predictions(model_lab)
    plt.savefig('predictions_lab.png')


if __name__ == '__main__':
    main()

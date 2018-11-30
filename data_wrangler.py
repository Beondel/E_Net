import numpy as np
from e_kernel import Kernel
import os
from pathlib import Path


# populates testing directory using MNIST master csv


def populate_testing():
    # load testing set into array
    img_array = np.genfromtxt("./MNIST/mnist_test.csv", delimiter=',')

    # save all testing images in 2d format
    for ix in range(10000):
        print(ix)
        # make a 2d matrix of a single line/image from the bigger array
        arr = np.zeros((28, 28), dtype=np.int16)
        for i in range(783):
            index = i + 1
            arr[index // 28][index % 28] = img_array[ix][index]
        np.savetxt("./MNIST/testing/" + str(ix) + ".csv", arr, fmt='%d', delimiter=',')


# populates training directory using MNIST master csv


def populate_training():
    # load training set into array
    img_array = np.genfromtxt("./MNIST/mnist_train.csv", delimiter=',')

    # save all training images in 2d format
    for ix in range(60000):
        if ix % 1000 == 0:
            print(ix)
        # make a 2d matrix of a single line/image from the bigger array
        arr = np.zeros((28, 28), dtype=np.int16)
        for i in range(783):
            index = i + 1
            arr[index // 28][index % 28] = img_array[ix][index]
        np.savetxt("./MNIST/training/" + str(ix) + ".csv", arr, fmt='%d', delimiter=',')


# populates directories specified by "dir" with entropy landscapes created by "kernel"
# kernel: Kernel object
# dir: string which specifies directory relative to ./MNIST/
def populate_dir(kernel, dir):
    for filename in os.listdir("./MNIST/" + dir):
        print(filename)
        arr = np.genfromtxt("./MNIST/" + dir + "/" + filename, dtype=np.int16, delimiter=',')
        ent_size = len(arr[0]) - 1
        ent_array = np.zeros((ent_size, ent_size), dtype=np.float)
        for i in range(ent_size):
            for j in range(ent_size):
                sub_array = np.array([[arr[i][j], arr[i][j + 1]],
                                      [arr[i + 1][j], arr[i + 1][j + 1]]])
                ent_array[i][j] = kernel.entropic_val(sub_array)
        np.savetxt("./MNIST/e_" + dir + "/" + filename, ent_array, fmt='%f', delimiter=',')


def populate_label_file(f):
    print("beginning " + f)
    img_array = np.genfromtxt("./MNIST/mnist_" + f + ".csv", delimiter=',')
    size = img_array.shape[0]
    label_array = np.zeros(size, dtype=np.int16)
    for i in range(size):
        label_array[i] = img_array[i][0]
    print(label_array)
    np.savetxt("./MNIST/" + f + "_labels.csv", label_array, fmt='%d', newline=',')


def merge_directories():
    # csv's
    test = np.genfromtxt("./MNIST/test_labels.csv", delimiter=',')
    train = np.genfromtxt("./MNIST/train_labels.csv", delimiter=',')
    labels = np.append(train, test)
    np.savetxt("./MNIST/labels.csv", labels, fmt='%d', delimiter=',')

    # images
    for i in range(60000):
        os.rename("./MNIST/training/" + str(i) + ".csv", "./MNIST/images/" + str(i) + ".csv")

    for i in range(10000):
        newName = str(i + 60000) + ".csv"
        os.rename("./MNIST/testing/" + str(i) + ".csv", "./MNIST/images/" + newName)


def separate_directories():
    for i in range(10000):
        newName = str(i + 60000) + ".csv"
        os.rename("./MNIST/images/" + newName, "./MNIST/testing/" + str(i) + ".csv")

    for i in range(60000):
        os.rename("./MNIST/images/" + str(i) + ".csv", "./MNIST/training/" + str(i) + ".csv")


if __name__ == "__main__":
    # comment out method calls as needed - these each take a while

    # populate_testing()
    # populate_training()

    # kernel = Kernel(2)
    # populate_dir(kernel, "testing")
    # populate_dir(kernel, "training")

    # populate_label_file("test")
    # populate_label_file("train")

    # merge_directories()
    # separate_directories()

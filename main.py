from builders import *
from sklearn import preprocessing
from sklearn import svm
import sys

DATA = 0
LABELS = 1

def accuracy(predicted, actual):
    return (predicted == actual).mean()

def lines_from_file(filename):
    file = open(filename, "r")
    lines = file.readlines()
    lines = [line[:-1] for line in lines]
    lines = [line.split('\t')[1] for line in lines]

    return lines

def read_data_set(samples_path, labels_path):
    samples = [line.split() for line in lines_from_file(samples_path)]
    labels  = lines_from_file(labels_path)

    return [samples, labels]

def main(argv):
    argc = len(argv)

    train_paths = [argv[1], argv[2]]
    test_paths  = [argv[3], argv[4]]
    
    train_set = read_data_set(train_paths[DATA], train_paths[LABELS])
    test_set = read_data_set(test_paths[DATA], test_paths[LABELS])

    train_set[DATA] = train_set[DATA][0:3000]
    train_set[LABELS] = train_set[LABELS][0:3000]

    builder = FeaturesFromWords()
    builder.make_vocabulary(train_set[DATA])
    print("Dictionary size: {}".format(len(builder.dict)))

    x_train = builder.get_features(train_set[DATA])
    x_test = builder.get_features(test_set[DATA])

    print("Got features")

    classifier = svm.SVC(C=10,kernel='linear')
    classifier.fit(x_train, train_set[LABELS])
    print("Model is fit")

    predicted = classifier.predict(x_test)
    print("score: ", accuracy(predicted, test_set[LABELS]))

    return

if __name__ == "__main__":
    main(sys.argv)

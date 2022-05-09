from builders import *
from classifiers import *
from sklearn import preprocessing
from sklearn import svm
import statistics
import sys

DATA = 0
LABELS = 1

def fprint(fmt, *args):
    print(fmt.format(*args))

def fprinterr(fmt, *args):
    print(fmt.format(*args), file=sys.stderr)

def accuracy(predicted, actual):
    print(actual)
    return (predicted == actual).mean()

def lines_from_file(filename):
    file = open(filename, "r")
    lines = file.readlines()
    lines = [line[:-1] for line in lines]
    lines = [line.split('\t')[1] for line in lines]

    return lines

def get_order(filename):
    file = open(filename, "r")
    lines = file.readlines()
    lines = [line[:-1] for line in lines]
    lines = [line.split('\t')[0] for line in lines]

    return lines

def main(argv):
    argc = len(argv)

    train_samples = [line.split() for line in lines_from_file(argv[1])]
    train_labels  = np.array(list(map(int, lines_from_file(argv[2]))))

    train_samples = train_samples
    train_labels  = train_labels

    test_samples  = [line.split() for line in lines_from_file(argv[3])]
    test_labels   = None

    if argc == 5:
        test_labels = np.array(list(map(int, lines_from_file(argv[4]))))

    builder = FeaturesFromChars()
    builder.make_vocabulary(train_samples)
    fprinterr("dictionary size: {}", len(builder.dict))

    x_train = builder.get_features(train_samples)
    x_test = builder.get_features(test_samples)

    fprinterr("train/test features: done")

    #classifier = svm.SVC(C=10,kernel='linear')
    #classifier.fit(x_train, train_labels)
    classifier = Knn_classifier(x_train, train_labels)
    fprinterr("fit(): done")

    predicted = classifier.predict(x_test)
    if test_labels is not None:
        fprinterr("score: {}", accuracy(predicted, test_labels))

    for a, b in zip(get_order(argv[3]), predicted):
        fprint("{},{}", a, b);


if __name__ == "__main__":
    main(sys.argv)

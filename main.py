import numpy as np
import sys

DATA = 0
LABELS = 1

def lines_from_file(filename):
    file = open(filename, "r")
    lines = file.readlines()
    lines = [line[:-1] for line in lines]
    lines = [line.split('\t') for line in lines]

    return lines

def main(argv):
    argc = len(argv)

    train_paths = [argv[1], argv[2]]
    test_paths  = [argv[3], argv[4]]


    return

if __name__ == "__main__":
    print(lines_from_file("f.txt"))

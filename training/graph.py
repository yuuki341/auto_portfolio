import numpy as np
import matplotlib.pyplot as plt
import argparse

def main(args):
    sample = np.loadtxt(f'../sample_1000/mcdropout_day{args.d}.csv', delimiter=',')
    plt.hist(sample[args.t])
    plt.savefig("sample.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int,default=0)
    parser.add_argument('-t', type=int,default=0)
    args = parser.parse_args()
    main(args)
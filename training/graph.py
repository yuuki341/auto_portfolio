import numpy as np
import matplotlib.pyplot as plt

days=0
sample = np.loadtxt(f'../sample_08/mcdropout_day{days}.csv', delimiter=',')

plt.hist(sample[547])

plt.savefig("sample.png")
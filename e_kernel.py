import numpy as np
import math


class Kernel:

    def __init__(self, b):
        self.b = b

    def entropic_val(self, arr):
        arr = np.reshape(arr, (arr.size))
        for i in range(len(arr)):
            arr[i] = math.ceil(arr[i])
        counts = {}
        for i in arr:
            if not i in counts.keys():
                counts[i] = 0
            counts[i] = counts[i] + 1
        entropy_sum = 0
        for i in counts.keys():
            counts[i] = counts[i] / len(arr)
            entropy_sum -= counts[i] * math.log(counts[i], self.b)

        return entropy_sum

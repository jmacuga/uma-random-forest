import math


def entropy_func(class_count, num_samples):
    return -class_count / num_samples * math.log(class_count / num_samples, 2)

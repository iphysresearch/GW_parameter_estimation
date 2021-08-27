import os
import fnmatch
import numpy as np
from scipy import stats
import csv


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def print_dict(d: dict,  ncol: int, prex: str = ''):
    temp = [f"{k}: {v}" for k, v in d.items()]
    print(prex+('\n'+prex).join([', '.join(temp[i:: ncol]) for i in range(ncol)]))


def ffname(path, pat):
    """Return a list containing the names of the files in the directory matches PATTERN.

    'path' can be specified as either str, bytes, or a path-like object.  If path is bytes,
        the filenames returned will also be bytes; in all other circumstances
        the filenames returned will be str.
    If path is None, uses the path='.'.
    'Patterns' are Unix shell style:
        *       matches everything
        ?       matches any single character
        [seq]   matches any character in seq
        [!seq]  matches any char not in seq
    Ref: https://stackoverflow.com/questions/33743816/how-to-find-a-filename-that-contains-a-given-string
    """
    return [filename
            for filename in os.listdir(path)
            if fnmatch.fnmatch(filename, pat)]


def writer_row(p, filename, mode, contentrow):
    with open(p / filename, mode) as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(contentrow)


def kl_divergence(samples, kde=stats.gaussian_kde, decimal=5, base=2.0):
    try:
        kernel = [kde(i, bw_method='scott') for i in samples]
    except np.linalg.LinAlgError:
        return float("nan")

    x = np.linspace(
        np.min([np.min(i) for i in samples]),
        np.max([np.max(i) for i in samples]),
        100
    )
    factor = 1.0e-5

    a, b = [k(x) for k in kernel]

    for index in range(len(a)):
        a[index] = max(a[index], max(a) * factor)
    for index in range(len(b)):
        b[index] = max(b[index], max(b) * factor)

    a = np.asarray(a)
    b = np.asarray(b)
    return stats.entropy(a, qk=b, base=base)


def js_divergence(samples, kde=stats.gaussian_kde, decimal=5, base=2.0):
    try:
        kernel = [kde(i) for i in samples]
    except np.linalg.LinAlgError:
        return float("nan")

    x = np.linspace(
        np.min([np.min(i) for i in samples]),
        np.max([np.max(i) for i in samples]),
        100
    )

    a, b = [k(x) for k in kernel]
    a = np.asarray(a)
    b = np.asarray(b)

    m = 1. / 2 * (a + b)
    kl_forward = stats.entropy(a, qk=m, base=base)
    kl_backward = stats.entropy(b, qk=m, base=base)
    return np.round(kl_forward / 2. + kl_backward / 2., decimal)


class MultipleOptimizer(object):
    """
    Learned from
    https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/7
    """
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        return [op.state_dict() for op in self.optimizers]


class MultipleScheduler(object):
    def __init__(self, *scheduler):
        self.schedulers = scheduler

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self):
        return [scheduler.state_dict() for scheduler in self.schedulers]

import multiprocessing
import traceback

import numpy as np
from numba import njit
from tqdm import tqdm

ONEOVERSQRT2PI = 1.0 / np.sqrt(2 * np.pi)


@njit(error_model='numpy', boundscheck=False, cache=True)
def cal_gauss(sigma, mu, x):
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return ONEOVERSQRT2PI * np.exp(exponent) / sigma


def _process_wrapper(args):
    func, single_args, use_traceback = args
    try:
        return func(*single_args)
    except Exception:
        if use_traceback:
            traceback.print_exc()
        return None


def batch_process(func, args_list, workers=1, use_tqdm=True, use_traceback=False):
    workers = max(workers, 1)

    if workers == 1:
        results = []
        for args in (tqdm(args_list) if use_tqdm else args_list):
            try:
                result = func(*args)
                if result is not None:
                    results.append(result)
            except Exception:
                if use_traceback:
                    traceback.print_exc()
        return results

    with multiprocessing.Pool(processes=workers) as pool:
        tasks = [(func, args, use_traceback) for args in args_list]
        results = []

        for result in (tqdm(pool.imap_unordered(_process_wrapper, tasks), total=len(tasks))
        if use_tqdm else pool.imap_unordered(_process_wrapper, tasks)):
            if result is not None:
                results.append(result)

        return results

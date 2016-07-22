import glob
import random
import os.path as op
import cPickle as pickle


SAVE_PARAMS_EVERY = 1000


def load_saved_params():
    st = 0
    for f in glob.glob("./cache/saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        with open("./cache/saved_params_%d.npy" % st, "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None


def save_params(iter, params):
    with open("./cache/saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY=10):
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    expcost = None

    for iter in xrange(start_iter + 1, iterations + 1):
        cost = None
        cost, grad = f(x)
        x -= step * grad

        x = postprocessing(x)

        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                # why this? 0.95 weight make more sense
                expcost = .95 * expcost + .05 * cost
            print "iter %d: %f" % (iter, expcost)

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x


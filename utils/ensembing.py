import numpy
from scipy.stats import stats


def posteriors_to_classes(posteriors):
    if len(posteriors.shape) > 1 and posteriors.shape[1] > 1:
        predicted = numpy.argmax(posteriors, 1)
    else:
        predicted = numpy.clip(numpy.sign(posteriors), a_min=0,
                               a_max=None)
    return predicted.astype(int)


def ensemble_posteriors(posteriors):
    avg_posteriors = numpy.mean(numpy.stack(posteriors, axis=0), axis=0)
    return avg_posteriors


def ensemble_voting(predictions):
    stacked = numpy.stack(predictions, axis=0)
    modals = stats.mode(stacked, axis=0)[0].squeeze().astype(int)
    return modals

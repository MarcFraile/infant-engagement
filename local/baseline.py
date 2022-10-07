def binary_accuracy(empirical_probability: float) -> float:
    """
    Given the `empirical_probability` of the positive class in a binary classification problem,
    return the theoretical best accuracy of a random classifier.

    * In mathematical terms, if both the classifier and the true class are modelled as independent random variables, what is the best achievable accuracy?
    * Achieved by always predicting the most likely class.
    * `empirical_probability` and return value expressed as fractions in range [0, 1].
    """

    return max(empirical_probability, 1 - empirical_probability)


def binary_f1(empirical_probability: float) -> float:
    """
    Given the `empirical_probability` of the positive class in a binary classification problem,
    return the theoretical best F1 score of a random classifier.

    * In mathematical terms, if both the classifier and the true class are modelled as independent random variables, what is the best achievable F1 score?
    * Achieved by always predicting the positive class.
    * `empirical_probability` and return value expressed as fractions in range [0, 1].
    """

    return (2 * empirical_probability) / (1 + empirical_probability)

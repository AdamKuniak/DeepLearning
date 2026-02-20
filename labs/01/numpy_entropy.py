#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    with open(args.data_path, "r") as data:
        data_dict = {}
        for line in data:
            line = line.rstrip("\n")
            if line in data_dict:
                data_dict[line] += 1
            else:
                data_dict[line] = 1
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
    data_distribution = {word : count / sum(data_dict.values()) for word, count in data_dict.items()}
    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.

    # TODO: Load model distribution, each line `string \t probability`.
    with open(args.model_path, "r") as model:
        model_dict = {}
        for line in model:
            line = line.rstrip("\n").split("\t")
            model_dict[line[0]] = line[1]
            # TODO: Process the line, aggregating using Python data structures.

    # TODO: Create a NumPy array containing the model distribution.
    shared_keys = list(data_distribution.keys() & model_dict.keys())
    # only interested in those words that are in both data and model
    data_shared_probs = np.array([data_distribution[word] for word in shared_keys], dtype=float)
    model_shared_probs = np.array([model_dict[word] for word in shared_keys], dtype=float)
    #print(model_probs.shape)
    #print(data_probs.shape)
    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    data_probs = np.array(list(data_distribution.values()), dtype=float)
    entropy = -np.sum(data_probs * np.log(data_probs))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    crossentropy = (-np.sum(data_shared_probs * np.log(model_shared_probs)) if len(shared_keys) == len(data_distribution) else np.inf)
    # data_distribution and shared keys must be the same length, meaning all data from data_distribution are in model, otherwise infinity
    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = crossentropy - entropy if crossentropy != np.inf else np.inf

    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(main_args)
    print(f"Entropy: {entropy:.2f} nats")
    print(f"Crossentropy: {crossentropy:.2f} nats")
    print(f"KL divergence: {kl_divergence:.2f} nats")

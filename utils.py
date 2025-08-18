import numpy as np
import pandas as pd
import warnings
import re


# Generator functions
def index_reconstructor(pred_matrix):
    return np.argmax(pred_matrix, axis=0)


def index_sampler(pred_matrix):
    # Convert to numpy array if it's a DataFrame
    probabilities = np.asarray(pred_matrix, dtype=np.float64)

    # Normalize probabilities column-wise
    probabilities /= probabilities.sum(axis=0, keepdims=True)

    # Compute cumulative sums for vectorized sampling
    cumulative_probs = np.cumsum(probabilities, axis=0)

    # Generate uniform random numbers for sampling
    random_values = np.random.rand(probabilities.shape[1])

    # Vectorized sampling of indices
    indices = (cumulative_probs < random_values).sum(axis=0)

    return indices

# Define the translation mapping from R to Python
translation_map = {
    '\\^': '**',
    'log10': 'np.log10',
    'log': 'np.log',
    'sqrt': 'np.sqrt',
    'exp': 'np.exp',
    'sin': 'np.sin',
    'sinh': 'np.sinh',
    'cos': 'np.cos',
    'cosh': 'np.cosh',
    'tan': 'np.tan',
    'tanh': 'np.tanh',
    'atan': 'np.arctan',
    'abs': 'np.abs'
}

# Helper function for translating R expressions to Python
def translate_r_to_python(expr):
    for r_func, py_func in translation_map.items():
        expr = re.sub(rf'\\b{r_func}\\b', py_func, expr)
    return expr


# Helper function that evaluates a dynamically generated expression
def tf_evaluation(predicted_tf, variables):
    predicted_tf = translate_r_to_python(predicted_tf)
    var_def = "\n".join([f"{var} = 1.0" for var in variables])
    full_eval_fun = f"{var_def}\n{predicted_tf}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            # Define function dynamically
            local_namespace = {}
            exec(f"def f_test():\n  {full_eval_fun.replace(chr(10), chr(10)+'  ')}", {}, local_namespace)
            tf_result = local_namespace['f_test']()
            result = True
        except Exception:
            result = None
    return result


variables = [
    "dem", "aspect", "slope", "bd", "sand", "clay",  # standard variables
    "lai", "mat_range", "map", "mat",  # new variables
    "ThetaS", "KSat", "vGenu_n"  # mHM variables
]


# Main function for index prediction
def generate_function_from_softmax(softmax_pred, vocab_df, variables=variables):

    # vocab to list
    vocab_df.loc[vocab_df.iloc[:, 0] == 0, vocab_df.columns[1]] = ""
    vocab = vocab_df.iloc[:, 1].tolist()

    # Transform log softmax to probabilities
    softmax_probabilities = np.exp(softmax_pred)

    # Get valid TF using reconstruction
    index_prediction = index_reconstructor(softmax_probabilities)
    ini_predicted_tf = "".join([vocab[i] for i in index_prediction])

    tf_eval = tf_evaluation(ini_predicted_tf, variables)
    fail_count = 0

    # Random sampling until valid function is generated
    while tf_eval is None and fail_count < 5000:
        fail_count += 1
        index_prediction = index_sampler(softmax_probabilities)
        predicted_tf = "".join([vocab[i] for i in index_prediction])
        tf_eval = tf_evaluation(predicted_tf, variables)

    if tf_eval is None:
        return ini_predicted_tf

    if fail_count == 0:
        predicted_tf = ini_predicted_tf

    # Fix occurrences of "++" and "--"
    for _ in range(5):
        predicted_tf = predicted_tf.replace("--", "-").replace("++", "-")

    return predicted_tf
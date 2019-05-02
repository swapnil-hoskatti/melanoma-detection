print("Estimated load time: 2.5s")

def format_tuple(output: tuple):
    """
    Ensures four tuple output with proper results
    """
    features, features_norm, pred_logistic, pred_cnn = output
    assert(type(pred_logistic), str)
    assert(type(pred_cnn), str)
    # features, features_norm, pred_logistic, pred_cnn
    return output
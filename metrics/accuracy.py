import numpy as np


class Accuracy:
    def __init__(self):
        pass
    
    def __call__(self, predictions, targets):
        assert (len(predictions) == len(targets)) and (len(predictions) > 0)
        
        predictions = np.array(predictions)
        targets = np.array(targets)                   

        corrects = sum(predictions == targets)
        n = len(predictions)
        
        return corrects / n
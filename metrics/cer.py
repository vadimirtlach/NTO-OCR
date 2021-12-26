import editdistance


class CER:
    def __init__(self, eps=1e-7):
        """
        eps:int - small value for avoiding divising by zero in denominator. default: 1e-7
        """
        
        self.eps = eps
        
    def __call__(self, predictions, targets):
        numerator = []
        denominator = []
        
        for prediction, target in zip(predictions, targets):
            levenshtein_distance = editdistance.eval(prediction, target)
            numerator.append(levenshtein_distance)
            denominator.append(len(target))
            
        return sum(numerator) / (sum(denominator) + self.eps)
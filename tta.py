# TODO
class TTA:
    def __init__(self, model, transforms, device="cpu"):
        self.model = model
        self.transforms = transforms
        self.device = device

    def predict(self, x):
        pass
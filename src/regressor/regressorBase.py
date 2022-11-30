
class Regressor:
    def __init__(self) -> None:
        self._model = None
    
    def train(self, data):
        raise NotImplementedError

    def evaluate(self, data):
        raise NotImplementedError
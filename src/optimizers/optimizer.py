
class Optimizer:
    def __init__(self, bitSpace, model, qmnAnalysis) -> None:
        self.bitSpace = bitSpace
        self.model = model
        self.qmnAnalysis = qmnAnalysis
    
    def optimize(self) -> None:
        return NotImplementedError
    

from src.regressor.regressorBase import Regressor
from sklearn.ensemble import RandomForestRegressor

class RFRegressor(Regressor):
    def __init__(self,n_estimators,max_features) -> None:
        self._model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=0)
    
    def train(self, data):
        self._model.train(data)
    
    def evaluate(self, data):
        self._model.evaluate(data)

class QmnAnalyzer:

    def __init__(self) -> None:
        self._computed = False
        self.__stats = dict()
        self.model = None

    def load_data():
        raise NotImplementedError
    
    def compute_stats():
        raise NotImplementedError

    @property
    def stats(self):
        assert self._computed,'First call compute_stats!'
        return self.__stats

    
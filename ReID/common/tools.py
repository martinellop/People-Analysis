
class ResultsDict(dict):
    def __setitem__(self, key, value):
        try:
            super().__setitem__(key, value) 
        except KeyError:
            self[key] = []
            super().__setitem__(key, value)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            self[key] = []
            return super().__getitem__(key)
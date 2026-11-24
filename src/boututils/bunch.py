# what we need from bunch


class Bunch:
    def __init__(self, **dict):
        for k, v in dict.items():
            setattr(self, k, v)

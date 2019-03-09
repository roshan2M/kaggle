from sklearn.feature_extraction.text import CountVectorizer

class CommentVectorizer:
    def __init__(self):
        self._vectorizers = []

    def get_count_vectorizers(self, max_features: int=1000, ngram_range: tuple=(1,2),
                              stop_words='english', binary: bool=True) -> int:
        self._vectorizers.append(CountVectorizer(max_features=max_features, ngram_range=ngram_range,
                                                 stop_words=stop_words, binary=binary))
        return len(self._vectorizers) - 1

    def fit(self, comments: list, index: int):
        self.validate_index(index)
        self._vectorizers[index].fit(comments)

    def transform(self, comments: list, index: int) -> list:
        self.validate_index(index)
        return self._vectorizers[index].transform(comments)

    def validate_index(self, index) -> bool:
        if index < 0 or index >= len(self._vectorizers):
            raise Exception('Index out of bounds.')
        elif self._vectorizers[index] is None:
            raise Exception('Vectorizer not initialized.')
        return True

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')


class FeatureExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract(self, text: str) -> list[str]:
        # Tokenize the text
        words = word_tokenize(text.lower())

        # Remove stopwords and non-alphabetic words
        filtered_words = [word for word in words if word.isalpha() and word not in self.stop_words]

        # Get the frequency distribution of the words
        freq_dist = FreqDist(filtered_words)

        num_common = max(1, len(filtered_words))

        # Extract the most common keywords
        common_keywords = [word for word, _ in freq_dist.most_common(num_common)]

        return common_keywords


if __name__ == '__main__':
    extractor = FeatureExtractor()
    description = "Can you build me a pipeline that will get a dataset from the following MySQL database: host -> 62.72.21.79, port -> 5432, database -> postgres, table -> iris, username -> postgres, password -> postgres. At the end exported it back as a CSV with name iris.csv."
    features = extractor.extract(description)
    print(features)

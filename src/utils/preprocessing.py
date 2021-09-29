import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
english_stopwords = stopwords.words('english')


def process_text(text: str) -> str:
    """ Remove all number, special characters, single letters, etc."""
    text = text.replace('-', ' ')
    processed_words = []
    for w in text.split(' '):
        w = ''.join(c for c in w if c.isalpha())
        w = w.lower()
        if w not in english_stopwords and len(w) > 1:
            processed_words.append(w)
    return ' '.join(processed_words)

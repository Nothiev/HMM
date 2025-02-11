
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

text = "Ecole Hexagone is providing NLP technique within the AI master courses."

tokens = nltk.word_tokenize(text)

print(tokens)


def compute_tf(text):
    text = text.lower()

    tokens = text.split()

    word_counts = {}
    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1

    total_words = len(tokens)

    # 5) Calcule TF pour chaque mot
    tf_dict = {}
    for word, count in word_counts.items():
        tf_dict[word] = count / total_words

    return tf_dict

def compute_IDF(tokens):
    




            


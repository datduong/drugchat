import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from smiles2graph import smiles2graph


def main():
    with open("../data/chebi.json", "rt") as f:
        js = json.load(f)

    # words = sum((text.lower().split() for text, _ in js), [])
    # vocab = list(set(words))
    # word_vec = CountVectorizer(vocabulary=vocab)

    corpus = [text for text, _ in js]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    data = []
    n = 0
    for i, (text, smi) in enumerate(js):
        try:
            graph = smiles2graph(smi)
        except:
            print(smi)
            n += 1
        vec = X[i]
        data.append({"word_vec": vec, "graph": graph})

    print("skip", n)
    with open("../data/chebi_0.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()


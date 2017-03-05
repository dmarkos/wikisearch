""" Hierarchical clustering of WikiPedia data using Ward variance minimization."""
import argparse
import joblib
import itertools
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from toolset import tokenizer
from toolset import Corpus


def cluster(corpus):
    """ Applies the hierarchical clustering and returns cluster model."""
    # Initialize the vectorizer.
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=3, max_features=10000,
                                 use_idf=True, stop_words='english',
                                 tokenizer=tokenizer, ngram_range=(1, 3))
    # Generate tf/idf matrix.
    print('Computing Tf/Idf matrix')
    tfidf_matrix = vectorizer.fit_transform(corpus.document_generator())
    print('Saving Tf/Idf matrix')
    joblib.dump(tfidf_matrix, 'tfidf.pkl')

    print('Applying clustering')
    # Generate linkage matrix.
    clusterer = AgglomerativeClustering(n_clusters=2, linkage='ward')
    cluster_model = clusterer.fit(tfidf_matrix.toarray())
    print('Saving cluster model')
    joblib.dump(cluster_model, 'model.pkl')

    # Display results
    docs = list(corpus.document_generator())
    titles = [doc.split('\n')[1] for doc in docs]
    ii = itertools.count()
    merges = [{'node_id': next(ii), 'left': x[0], 
               'right':x[1]} for x in cluster_model.children_]
    print(merges)

def main():
    """ Main function of the module."""
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='Path to text document collection')
    args = parser.parse_args()

    cluster(Corpus(args.text, False, None))

if __name__ == '__main__':
    main()

""" A set of tools to be used for wikipedia document clustering. """

from __future__ import print_function

import os
import sys
import shutil
import itertools
import re
import time
from optparse import OptionParser
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer

def tokenize(text):
    """ Takes a String as input and returns a list of its tokens. """

    filtered_tokens = []
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in
              nltk.word_tokenize(sent)]
    # Remove tokens that do not contain letters.
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def stem(tokens):
    """ Takes a list of tokens as input and stems each entry. """
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(token) for token in tokens]

def tokenizer(text):
    """ Tokenizes and then stems a given text. """
    return stem(tokenize(text))


class Corpus(object):
    """
    A Corpus object is initiated on a collection of documents extracted from a
    wikipedia dump and provides methods for handling the documents.
    """

    def __init__(self, corpus_file_path, format_corpus, sub_size):
        self.corpus_file_path = corpus_file_path
        self.formatted_corpus_file_path = os.getcwd() + "/formatted"
        self.format_corpus = format_corpus
        self.sub_size = sub_size
        # Format the corpus if it's not already formatted.
        if self.format_corpus:
            self.format()
        self.document_paths = self.get_document_paths()

    def get_document_paths(self):
        """ Returns a list of the filepaths to all the documents in the corpus."""
        document_paths = []
        for document_folder in os.listdir(self.formatted_corpus_file_path):
            for document_file in os.listdir(self.formatted_corpus_file_path + '/'
                                            + document_folder):
                document_paths.append(self.formatted_corpus_file_path
                                      + '/' + document_folder + '/' + document_file)
        return document_paths

    def format(self):
        """ Change the format of the corpus file to one document per file."""
        print('Formatting corpus\n')
        # If a formatted directory already exists, overwrite it.
        if os.path.exists(self.formatted_corpus_file_path):
            shutil.rmtree(self.formatted_corpus_file_path)
            os.makedirs(self.formatted_corpus_file_path)
        n_docs = 0

        for document_folder in os.listdir(self.corpus_file_path):
            os.makedirs(self.formatted_corpus_file_path + '/' + document_folder)
            for document_file in os.listdir(self.corpus_file_path + '/' + document_folder):
                # The document's XML like format does not have a root element so it
                # needs to be added in order for the ElementTree to be created.
                with open(self.corpus_file_path + '/' + document_folder + '/'
                          + document_file) as document_file_content:
                    # Escape all lines except <doc> tag lines to avoid XML parsing
                    # errors
                    document_file_content_escaped = []
                    for line in document_file_content.readlines():
                        if (not line.startswith('<doc id')
                                and not line.startswith('</doc>')):
                            document_file_content_escaped.append(escape(line))
                        else:
                            document_file_content_escaped.append(line)

                    document_file_iterator = \
                            itertools.chain('<root>', document_file_content_escaped, '</root>')
                    # Parse the document file using the iterable.
                    documents = ET.fromstringlist(document_file_iterator)
                # Each document file contains multiple documents each wrapped in a
                # doc tag
                for i, doc in enumerate(documents.findall("doc")):
                    # Save each document in a separate file.
                    filepath = '/'.join([self.formatted_corpus_file_path, document_folder,
                                         document_file + '_' + str(i)])
                    with open(filepath, 'w+') as output_document_file:
                        output_document_file.write(doc.text)
                    # If a subcollection size has been specified, stop when it is
                    # reached
                    n_docs += 1
                    if self.sub_size != None and n_docs >= self.sub_size:
                        return 0

    def document_generator(self):
        """
        Yields the documents of the corpus in String form.
        """
        for path in self.document_paths:
            with open(path) as document_file_content:
                yield document_file_content.read()[:2000]

    def get_vocabulary(self):
        """
        Returns a pandas Series data structure that matches stems to the tokens
        they derived from.
        """
        vocabulary_tokenized = []
        vocabulary_stemmed = []
        # Initiate a new document generator when this method is called.
        corpus = self.document_generator()

        for document in corpus:
            document_tokens = tokenize(document)
            vocabulary_tokenized.extend(document_tokens)
            # Remove duplicate tokens by casting to set and back to list.
            vocabulary_tokenized = list(set(vocabulary_tokenized))
        vocabulary_stemmed = stem(vocabulary_tokenized)

        # Create pandas series that matched stems to tokens with stems as indeces.
        vocabulary = pd.Series(vocabulary_tokenized, index=vocabulary_stemmed)

        return vocabulary


class ClusterMaker(object):
    """
    Applies clustering on text data using the kmeans algorithm and pickles the kmeans
    model. Enables cluster information extraction and visualization.
    """
    def __init__(self, n_clusters, n_dimensions=None):
        self.n_clusters = n_clusters
        self.n_dimensions = n_dimensions

    def make(self, corpus):
        """
        Applies kmeans clustering on the corpus and returns the kmeans model.
        """
        start_time = time.time()
        print("DEBUG Making cluster model")
        # Initialize the vectorizer.
        vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=10000,
                                     use_idf=True, stop_words='english',
                                     tokenizer=tokenizer, ngram_range=(1, 3))
        print("DEBUG Created vectorizer")
        # Compute the Tf/Idf matrix of the corpus.
        tfidf_matrix = vectorizer.fit_transform(corpus.document_generator())
        print("DEBUG Computed tfidf")

        # Apply latent semantic analysis.
        if self.n_dimensions != None:
            print('Performing latent semantic analysis')
            svd = TruncatedSVD(self.n_dimensions)
            # Normalize SVD results for better clustering results.
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)
            tfidf_matrix = lsa.fit_transform(tfidf_matrix)
            print('DEBUG LSA completed')

        # Do the clustering.
        kmodel = KMeans(self.n_clusters, init='k-means++', n_init=1, max_iter=100,
                        verbose=True)
        print('Clustering with %s' % kmodel)
        kmodel.fit(tfidf_matrix)
        print("Cluster model created in %d'" % ((time.time()-start_time) / 60))
        joblib.dump(kmodel, 'kmodel.pkl')
        #  cluster_labels = kmodel.labels_
        #  cluster_centers = kmodel.cluster_centers_

        # Print some info.
        print("Top terms per cluster:")
        if self.n_dimensions != None:
            original_space_centroids = svd.inverse_transform(kmodel.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = kmodel.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in range(self.n_clusters):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
                print()

        return kmodel


def main():
    """ Method used for testing, it will be removed after module is completed. """
    # Configure option parsing.
    usage = "Usage: toolset.py <corpus> [options]"
    parser = OptionParser(usage)
    parser.add_option('--format', help='Format the corpus.',
                      dest='format_corpus', action='store_true', default=False,
                      metavar='BOOLEAN')
    parser.add_option('--format-sub', help='Format a subcollection of the corpus.',
                      dest='sub_size', action='store', type='int', metavar='INTEGER')
    parser.add_option('--lsa', help='Reduce dimensions using latent semantic analysis',
                      dest='n_dimensions', action='store', type='int')
    (options, args) = parser.parse_args()

    if not os.path.exists(args[0]):
        print('File does not exist.')
        sys.exit(1)

    corpus = Corpus(args[0], options.format_corpus or options.sub_size,
                    options.sub_size)
    cmaker = ClusterMaker(8, options.n_dimensions)
    kmodel = cmaker.make(corpus)
    # Dump the k-means model.
    joblib.dump(kmodel, 'km.pkl')

if __name__ == "__main__":
    main()


""" A set of tools to be used for wikipedia document clustering. """

from __future__ import print_function

import os
import sys
import shutil
import itertools
import re
import time
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.snowball import SnowballStemmer
import click


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """ Command line interface for toolset.py."""
    pass

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
    """ Initialized using a collection of documents extracted from a
        wikipedia dump and provides methods for handling the documents.
    """

    def __init__(self, corpus_file_path):
        self.corpus_file_path = corpus_file_path
        self.document_paths = self._get_document_paths()

    def _get_document_paths(self):
        """ Returns a list of the filepaths to all the documents in the corpus."""
        document_paths = []
        for document_folder in os.listdir(self.corpus_file_path):
            for document_file in os.listdir(self.corpus_file_path + '/'
                                            + document_folder):
                document_paths.append(self.corpus_file_path
                                      + '/' + document_folder + '/' + document_file)
        return document_paths

    def format(self, sub_size=None, output_file_path=os.getcwd() + "/formatted"):
        """ Change the format of the corpus file to one document per file."""
        # If a formatted collection directory already exists, overwrite it.
        if os.path.exists(output_file_path):
            shutil.rmtree(output_file_path)
            os.makedirs(output_file_path)
        n_docs = 0

        for document_folder in os.listdir(self.corpus_file_path):
            os.makedirs(output_file_path + '/' + document_folder)
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
                    filepath = '/'.join([output_file_path, document_folder,
                                         document_file + '_' + str(i)])
                    with open(filepath, 'w+') as output_document_file:
                        output_document_file.write(doc.text)
                    # If a subcollection size has been specified, stop when it is
                    # reached
                    n_docs += 1
                    if sub_size != None and n_docs >= sub_size:
                        return 0

    def document_generator(self):
        """
        Yields the documents of the corpus in String form.
        """
        for path in self.document_paths:
            with open(path) as document_file_content:
                yield document_file_content.read()

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

    def get_stats(self):
        """ Prints statistics about the corpus with a formatted corpus file as input."""
        start_time = time.time()
        n_docs = 0
        # Document size metrics expressed in number of features.
        doc_size = 0
        total_doc_size = 0
        max_doc_size = 0
        min_doc_size = 0
        avg_doc_size = 0
        for path in self.document_paths:
            with open(path) as document_file_content:
                doc_size = 0
                for line in document_file_content:
                    if not (line.startswith('<doc') or line.startswith('</doc>')):
                        doc_size += len(line.split(' '))
                # Update metric values
                if n_docs == 0:
                    min_doc_size = doc_size
                n_docs += 1
                print('Documents processed: ' + str(n_docs) +
                      ', Rate: ' + str(round(n_docs/(time.time()-start_time))) + 'docs/sec')
                total_doc_size += doc_size
                if doc_size > max_doc_size:
                    max_doc_size = doc_size
                if doc_size < min_doc_size:
                    min_doc_size = doc_size
        avg_doc_size = total_doc_size/n_docs
        print()
        print()
        print('Number of documents: ' + str(n_docs))
        print()
        print('Document size metrics: ')
        print('    Max: ' + str(max_doc_size))
        print('    Min: ' + str(min_doc_size))
        print('    Average: ' + str(avg_doc_size))

class ClusterMaker(object):
    """
    Applies clustering on text data using the kmeans algorithm and pickles the kmeans
    model. Enables cluster information extraction and visualization.
    """
    def __init__(self, n_clusters, n_dimensions=None):
        self.n_clusters = n_clusters
        self.n_dimensions = n_dimensions

    @staticmethod
    def extract_tfidf(corpus):
        """ Returns the Tf/Idf matrix of the corpus and pickles Tf/Idf matrix and feature list."""
        # Initialize the vectorizer.
        vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=10000,
                                     use_idf=True, stop_words='english',
                                     tokenizer=tokenizer, ngram_range=(1, 3))
        print("DEBUG Created vectorizer")
        # Compute the Tf/Idf matrix of the corpus.
        tfidf_matrix = vectorizer.fit_transform(corpus.document_generator())
        # Get feature names from the fitted vectorizer.
        features = vectorizer.get_feature_names()
        print(tfidf_matrix.shape)
        print("DEBUG Computed tfidf")
        joblib.dump(tfidf_matrix, 'tfidf.pkl')
        joblib.dump(features, 'features.pkl')
        return tfidf_matrix

    def kmeans(self, corpus=None, tfidf_path=None, verbose=False):
        """ Applies kmeans clustering on the corpus and returns the kmeans model."""
        print("DEBUG Making cluster model")

        # Compute or load Tf/Idf matrix.
        if tfidf_path is None:
            tfidf_matrix = self.extract_tfidf(corpus)
            print(tfidf_matrix.shape)
        else:
            tfidf_matrix = joblib.load('tfidf.pkl')
            print(tfidf_matrix.shape)
            print('Loaded Tf/Idf matrix.')

        # Apply latent semantic analysis.
        if self.n_dimensions != None:
            print('Performing latent semantic analysis')
            svd = TruncatedSVD(self.n_dimensions)
            # Normalize SVD results for better clustering results.
            lsa = make_pipeline(svd, Normalizer(copy=False))
            tfidf_matrix = lsa.fit_transform(tfidf_matrix)
            print(tfidf_matrix.shape)
            print('DEBUG LSA completed')

        # Do the clustering.
        start_time = time.time()
        layer1_kmodel = KMeans(n_clusters=1000, init='k-means++', n_init=1, max_iter=100,
                               verbose=True)
        layer2_kmodel = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=1, max_iter=100,
                               verbose=True)
        print('Clustering with %s' % layer1_kmodel)
        layer1_kmodel.fit(tfidf_matrix)
        layer2_kmodel.fit(layer1_kmodel.cluster_centers_)
        end_time = time.time()
        joblib.dump(layer2_kmodel, 'kmodel.pkl')
        #  cluster_labels = kmodel.labels_
        #  cluster_centers = kmodel.cluster_centers_

        if verbose:
            # Print some info.
            print("Top terms per cluster:")
            if self.n_dimensions != None:
                original_space_centroids = svd.inverse_transform(layer2_kmodel.cluster_centers_)
                order_centroids = original_space_centroids.argsort()[:, ::-1]
            else:
                order_centroids = layer2_kmodel.cluster_centers_.argsort()[:, ::-1]

            features = joblib.load('features.pkl')
            for i in range(self.n_clusters):
                print("Cluster %d:" % i, end='')
                for ind in order_centroids[i, :10]:
                    print(' %s' % features[ind], end='')
                    print()
        print('Clustering completed after ' + str(round((end_time-start_time)/60)) + "' "
              + str(round((end_time-start_time)%60)) + "''")
        return layer2_kmodel

    def hac(self, corpus=None, tfidf_path=None, verbose=False):
        """ Apply Hierarchical Agglomerative Clustering on text data."""
        # Compute or load Tf/Idf matrix.
        if tfidf_path is None:
            tfidf_matrix = self.extract_tfidf(corpus)
            print(tfidf_matrix.shape)
        else:
            tfidf_matrix = joblib.load('tfidf.pkl')
            print(tfidf_matrix.shape)
            print('Loaded Tf/Idf matrix.')

        # Apply latent semantic analysis.
        if self.n_dimensions != None:
            print('Performing latent semantic analysis')
            svd = TruncatedSVD(self.n_dimensions)
            # Normalize SVD results for better clustering results.
            lsa = make_pipeline(svd, Normalizer(copy=False))
            tfidf_matrix = lsa.fit_transform(tfidf_matrix)

            print(tfidf_matrix.shape)
            print('DEBUG LSA completed')


        # Calculate documente distance matrix from Tf/Idf matrix
        dist = 1 - cosine_similarity(tfidf_matrix)
        print('DEBUG Computed distance matrix.')

        start_time = time.time()
        # Generate HAC model.
        hac_model = AgglomerativeClustering(linkage='ward', n_clusters=self.n_clusters)
        # Fit the model on the distance matrix.
        hac_model.fit(dist)
        end_time = time.time()
        joblib.dump(hac_model, 'hac.pkl')
        print('DEBUG Generated HAC model.')

        if verbose:
            # Visualize cluster model
            children = hac_model.children_
            merges = [{'node_id': node_id+len(dist),
                       'right': children[node_id, 0], 'left': children[node_id, 1]
                      } for node_id in range(0, len(children))]
            for merge_entry in enumerate(merges):
                print(merge_entry[1])

        print('Clustering completed after ' + str(round((end_time-start_time)/60)) + "' "
              + str(round((end_time-start_time)%60)) + "''")
        return hac_model

@cli.command()
@click.argument('corpus')
@click.option('--sub_size', default=None, type=int,
              help='Set the number of formatted documents.')
@click.option('-o', default=None, type=str,
              help='Set output file path.')
def format_corpus(**kwargs):
    """ Command to format the corpus."""
    if not os.path.exists(os.path.abspath(kwargs['corpus'])):
        print(os.path.abspath(kwargs['corpus']) + ' does not exist.')
        sys.exit(1)

    corpus = Corpus(os.path.abspath(kwargs['corpus']))
    corpus.format(kwargs['sub_size'], os.path.abspath(kwargs['o']))

@cli.command()
@click.argument('corpus')
def extract_tfidf(**kwargs):
    """ Click command to extract Tf/Idf matrix."""
    if not os.path.exists(os.path.abspath(kwargs['corpus'])):
        print(os.path.abspath(kwargs['corpus']) + ' does not exist.')
        sys.exit(1)

    corpus = Corpus(os.path.abspath(kwargs['corpus']))
    ClusterMaker.extract_tfidf(corpus)

@cli.command()
@click.argument('corpus')
@click.option('--tfidf', default=None,
              help='Use a pre-computed Tf/Idf matrix')
@click.option('--n_dimensions', default=None, type=int,
              help='Reduce dimensions using Latent Semantic Analysis.')
@click.option('--n_clusters', default=10,
              help='Set number of clusters.')
@click.option('--verbose', is_flag=True, help='Print clustering information.')
def kmeans(**kwargs):
    """ Click command to apply kmeans clustering."""
    if not os.path.exists(os.path.abspath(kwargs['corpus'])):
        print(os.path.abspath(kwargs['corpus']) + ' does not exist.')
        sys.exit(1)

    corpus = Corpus(os.path.abspath(kwargs['corpus']))
    if kwargs['tfidf'] is None:
        cmaker = ClusterMaker(kwargs['n_clusters'], kwargs['n_dimensions'])
        kmodel = cmaker.kmeans(corpus=corpus, tfidf_path=kwargs['tfidf'],
                               verbose=kwargs['verbose'])
        # Dump the k-means model.
        joblib.dump(kmodel, 'km.pkl')
    else:
        cmaker = ClusterMaker(kwargs['n_clusters'], kwargs['n_dimensions'])
        kmodel = cmaker.kmeans(corpus=corpus, tfidf_path=kwargs['tfidf'],
                               verbose=kwargs['verbose'])
        # Dump the k-means model.
        joblib.dump(kmodel, 'km.pkl')

@cli.command()
@click.argument('corpus')
@click.option('--tfidf', default=None,
              help='Use a pre-computed Tf/Idf matrix')
@click.option('--n_dimensions', default=None, type=int,
              help='Reduce dimensions using Latent Semantic Analysis.')
@click.option('--n_clusters', default=10,
              help='Set number of clusters.')
@click.option('--verbose', is_flag=True, help='Print clustering information.')
def hac(**kwargs):
    """ Click command to apply kmeans clustering."""
    if not os.path.exists(os.path.abspath(kwargs['corpus'])):
        print(os.path.abspath(kwargs['corpus']) + ' does not exist.')
        sys.exit(1)

    corpus = Corpus(os.path.abspath(kwargs['corpus']))
    if kwargs['tfidf'] is None:
        cmaker = ClusterMaker(kwargs['n_clusters'], kwargs['n_dimensions'])
        hac_model = cmaker.hac(corpus=corpus, tfidf_path=kwargs['tfidf'],
                               verbose=kwargs['verbose'])
        # Dump the HAC model.
        joblib.dump(hac_model, 'km.pkl')
    else:
        cmaker = ClusterMaker(kwargs['n_clusters'], kwargs['n_dimensions'])
        hac_model = cmaker.hac(corpus=corpus, tfidf_path=kwargs['tfidf'],
                               verbose=kwargs['verbose'])
        # Dump the HAC model.
        joblib.dump(hac_model, 'km.pkl')

@cli.command()
@click.argument('corpus')
def stats(**kwargs):
    """ Command to print statistics."""
    if not os.path.exists(os.path.abspath(kwargs['corpus'])):
        print(os.path.abspath(kwargs['corpus']) + ' does not exist.')
        sys.exit(1)
    corpus = Corpus(os.path.abspath(kwargs['corpus']))
    corpus.get_stats()


if __name__ == "__main__":
    cli()


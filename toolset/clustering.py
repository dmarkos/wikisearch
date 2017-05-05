# -*- coding: utf-8 -*-
""" Provides methods for applying clustering on a text document collection.
"""
import joblib
import re
import time

import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


def tokenize(text):
    """ Takes a String as input and returns a list of its tokens.

        Args:
            text (str): A string object.

        Returns:
            filtered_tokens: A list of the tokens in the string after removing
                duplicates and tokens that contain only numbers.

    """
    filtered_tokens = []
    tokens = [
        word.lower()
        for sent in nltk.sent_tokenize(text)
        for word in nltk.word_tokenize(sent)
    ]
    # Remove tokens that do not contain letters.
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def stem(tokens):
    """ Takes a list of tokens as input and stems each entry.

        NLTK's SnowballStemmer is used for the stemming.

        Args:
            tokens (:list:'str'): A list of tokens.
        Returns:
            stems (:list:'str'): The list containing the stems of the tokens
            given as input.

    """
    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(token) for token in tokens]

    return stems


def tokenizer(text):
    """ Tokenizes and then stems a given text.

    Simply combines the tokenize() and stem() methods. This method is used
    by by the TfidfVectorizer for the calculation of the Tf/Idf matrix.

    Args:
        text (str): A string object.

    Returns:
        stems (:list:'str'): A list containing the stems of the input string.
    """
    stems = stem(tokenize(text))
    return stems


class ClusterMaker(object):
    """ Wrapper for quickly applying some clustering algorithms.

    Applies clustering using the kmeans or hac algorithm.

    Args:
        n_clusters (int): The number of clusters to be created.
        n_dimensions (int): When given a value, specifies the number of
            dimensions of the vector space after applying Latent Semantic
            Analysis. Defaults to None.

    Attributes:
        n_clusters (int): The number of clusters to be created.
        n_dimensions (int): When given a value, specifies the number of
        dimensions of the vector space after applying Latent Semantic
        Analysis. Defaults to None.
    """

    def __init__(self, corpus):
        self.corpus = corpus

    def extract_tfidf(self):
        """ Calculates the Tf/Idf matrix of the document collection.

        The Tf/Idf matrix is in sparse matrix format. After calculation,
        the matrix and the features of the collection are saved in files.

        Args:
            self.corpus (:obj:'Corpus'): The Corpus object of the document
                collection.

        Returns:
           tfidf (sparse matrix): The Tf/idf matrix of the document
               collection.

        """
        print('Constructing Tf/Idf matrix...')
        # Initialize the vectorizer.
        hasher = HashingVectorizer(n_features=10000, stop_words='english',
                                   non_negative=True, norm=None, binary=False)
        vactorizer = make_pipeline(hasher, TfidfTransformer())
        
        # Compute the Tf/Idf matrix of the corpus.
        tfidf = vectorizer.fit_transform(
            self.corpus.document_generator())

        # Get feature names from the fitted vectorizer.
        features = vectorizer.get_feature_names()
        joblib.dump(tfidf, 'tfidf.pkl')
        joblib.dump(features, 'features.pkl')

        return tfidf

    def kmeans(self,
               n_clusters,
               tfidf=None,
               n_dimensions=None,
               verbose=False):
        """ Applies kmeans clustering on a document collection.

        Args:
            self.corpus (:obj:'Corpus'): The Corpus object of the document
                collection. Defaults to None. Only used when no pre-computed
                Tf/Idf matrix is given.
            tfidf_path (str): The path to the file containing the Tf/Idf matrix
                .pkl file. Defaults to None and in this case the Tf/Idf matrix
                is calculated.
            verbose (bool): When True additional information will be printed.
                Defaults to False.

        Returns:
            kmodel (:obj:'Kmeans'): Scikit KMeans clustering model.

        """
        
        # Compute or load Tf/Idf matrix.
        if tfidf is None:
            tfidf = self.extract_tfidf(self.corpus)
            print(tfidf.shape)

        # Apply latent semantic analysis.
        if n_dimensions is not None:
            print('Performing latent semantic analysis...')
            svd = TruncatedSVD(n_dimensions)
            # Normalize SVD results for better clustering results.
            lsa = make_pipeline(svd, Normalizer(copy=False))
            tfidf = lsa.fit_transform(tfidf)
            print(tfidf.shape)

        # Do the clustering.
        start_time = time.time()
        print('Clustering...')
        kmodel = MiniBatchKMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=1,
            max_iter=10,
            verbose=True)
        kmodel.fit(tfidf)
        end_time = time.time()

        # Create a matching of the clusters and the ids of the documents
        # they contain.
        cluster_doc = pd.Series()
        for i in range(kmodel.n_clusters):
            ids = []
            for docid, cluster in enumerate(kmodel.labels_):
                if cluster == i:
                    ids.append(docid)
                    cluster_doc.loc[i] = ids

        if verbose:
            # Print some info.
            print("Top terms per cluster:")
            if n_dimensions is not None:
                original_space_centroids = svd.inverse_transform(
                    kmodel.cluster_centers_)
                order_centroids = original_space_centroids.argsort()[:, ::-1]
            else:
                order_centroids = kmodel.cluster_centers_.argsort()[:, ::-1]

            features = pickle.load(open('features.pkl', 'rb'))
            cluster_word = pd.Series()
            for i in range(n_clusters):
                cluster_features = []
                print("Cluster %d:" % i)
                for ind in order_centroids[i, :100]:
                    cluster_features.append(features[ind])
                cluster_word.loc[i] = cluster_features

        pickle.dump(kmodel, open('kmodel.pkl', 'wb'))
        pickle.dump(kmodel.cluster_centers_, open('centers.pkl', 'wb'))
        pickle.dump(cluster_doc, open('cluster_doc.pkl', 'wb'))
        pickle.dump(cluster_word, open('cluster_word.pkl', 'wb'))

        print('Clustering completed after ' +
              str(round((end_time - start_time) / 60)) + "' " +
              str(round((end_time - start_time) % 60)) + "''")

        return kmodel

    def hac(self,
            n_clusters,
            verbose=False,
            tfidf=None,
            n_dimensions=None):
        """ Apply Hierarchical Agglomerative Clustering on a document collection.

        This method generates a hierarchical clustering tree for the collection.
        The leaves of the tree are clusters consisting of single documents.
        The tree is then saved by saving the list of merges in a file.

        Each entry of this list contains the two tree nodes that were merged to
        create a new node and the new node's id. Node ids less than the number
        of leaves represent leaves, while node ids greater than the number of
        leaves indicate internal nodes.

        Args:
            self.corpus (:obj:'Corpus'): The Corpus object of the document
                collection. Defaults to None. Only used when no pre-computed
                Tf/Idf matrix is given.
            tfidf_path (str): The path to the file containing the Tf/Idf matrix
                .pkl file. Defaults to None and in this case the Tf/Idf matrix
                is calculated.
            verbose (bool): When True additional information will be printed.
                Defaults to False.

        Returns:
            hac_model (:obj:'AgglomerativeClustering'): The HAC model fitted on
            the document collection.

        """
        # Compute or load Tf/Idf matrix.
        if tfidf is None:
            tfidf = self.extract_tfidf(self.corpus)
            print(tfidf.shape)

        # Apply latent semantic analysis.
        if n_dimensions is not None:
            print('Performing latent semantic analysis')
            svd = TruncatedSVD(n_dimensions)
            # Normalize SVD results for better clustering results.
            lsa = make_pipeline(svd, Normalizer(copy=False))
            tfidf = lsa.fit_transform(tfidf)

            print(tfidf.shape)

        # Calculate documente distance matrix from Tf/Idf matrix
        print('Constructing distance matrix...')
        dist = 1 - cosine_similarity(tfidf)

        start_time = time.time()
        print('Clustering...')
        # Generate HAC model.
        hac_model = AgglomerativeClustering(
            linkage='ward', n_clusters=n_clusters)
        # Fit the model on the distance matrix.
        hac_model.fit(dist)
        end_time = time.time()
        pickle.dump(hac_model, open('hac.pkl', 'wb'))

        if verbose:
            # Visualize cluster model
            children = hac_model.children_
            merges = [{
                'node_id': node_id + len(dist),
                'right': children[node_id, 0],
                'left': children[node_id, 1]
            } for node_id in range(0, len(children))]
            pickle.dump(merges, open('merges.pkl', 'wb'))
            pickle.dump(children, open('children.pkl', 'wb'))

            for merge_entry in enumerate(merges):
                print(merge_entry[1])

        print('Clustering completed after ' +
              str(round((end_time - start_time) / 60)) + "' " +
              str(round((end_time - start_time) % 60)) + "''")
        return hac_model

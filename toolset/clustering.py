import pickle
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
        pickle.dump(tfidf_matrix, open('tfidf.pkl', 'wb'))
        pickle.dump(features, open('features.pkl', 'wb'))
        return tfidf_matrix

    def kmeans(self, corpus=None, tfidf_path=None, verbose=False):
        """ Applies kmeans clustering on the corpus and returns the kmeans model."""
        print("DEBUG Making cluster model")

        # Compute or load Tf/Idf matrix.
        if tfidf_path is None:
            tfidf_matrix = self.extract_tfidf(corpus)
            print(tfidf_matrix.shape)
        else:
            tfidf_matrix = pickle.load(open('tfidf.pkl', 'rb'))
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
        layer1_kmodel = KMeans(n_clusters=100, init='k-means++', n_init=1, max_iter=10,
                               verbose=True)
        layer2_kmodel = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=1, max_iter=10,
                               verbose=True)
        print('Clustering with %s' % layer1_kmodel)
        layer1_kmodel.fit(tfidf_matrix)
        print('Clustering with %s' % layer2_kmodel)
        layer2_kmodel.fit(layer1_kmodel.cluster_centers_)
        end_time = time.time()
        pickle.dump(layer1_kmodel, open('layer1_kmodel.pkl', 'wb'))
        pickle.dump(layer1_kmodel.cluster_centers_, open('centers.pkl', 'wb'))
        pickle.dump(layer2_kmodel, open('layer2_kmodel.pkl', 'wb'))
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
            tfidf_matrix = pickle.load(open('tfidf.pkl', 'rb'))
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
            joblib.dump(merges, 'merges.pkl')
            joblib.dump(children, 'children.pkl')

            for merge_entry in enumerate(merges):
                print(merge_entry[1])

        print('Clustering completed after ' + str(round((end_time-start_time)/60)) + "' "
              + str(round((end_time-start_time)%60)) + "''")
        return hac_model

import os
import sys
import shutil
import itertools
import re
import time
import random
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

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
                    # To pick documents at random from the whole collection,
                    # a document is picked with a possibility which depends on the
                    # size of the sub-collection.
                    pos = sub_size/5331608
                    if random.random() > pos:
                        continue
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

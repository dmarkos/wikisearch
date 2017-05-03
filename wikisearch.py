# -*- coding: utf-8 -*-
""" Performs a simulated Scatter/Gather browsing and extracts execution information."""

import math
import os
import shutil
import time
import re

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def scatter(hac_model, working_set, n_clusters):
    """ Scatter step of the Scatter/Gather browsing method.

    Args:
        gclusters (:list:'int'): The clusters selected in the gather step each cluster
            is a node of the hierarchical tree and is represented by its id.
    Returns:
        wset (:list:'int'): The new working set of clusters each one is a node of
            the hierarchical tree and is represented by its id.

    """

    children = hac_model.children_
    #  cluster_doc = pickle.load(open('cluster_doc.pkl', 'rb'))

    # Replace a cluster from the working set with its children until the number of clusters
    # in the working set is n_clusters.
    while len(working_set) < n_clusters:
        # Replace higher level clusters (nodes closer to the root) as they tend to be bigger
        # in size. The cluster with the largest id was created last during the tree creation.
        working_set = sorted(working_set)
        curr_cluster_id = working_set[-1]

        if curr_cluster_id < hac_model.n_leaves_:  # Only leaves remain in the working set.
            return working_set

        working_set.pop()
        working_set.extend(
            children[curr_cluster_id -
                     hac_model.n_leaves_])  # First 100 ids represent leaves.
        #  print([len(get_docs(cluster_id, hac_model, cluster_doc)) for cluster_id in working_set])

    return working_set


def gather(working_set):
    """ Simulates a user selecting half of the clusters each time.

    Args:
        working_set (:list:'int'): The ids of the clusters in the working set.
    Returns:
        new_working_set (:list:'int'): The ids of the selected clusters.

    """
    new_working_set = working_set[:math.floor(len(working_set) / 2)]

    return new_working_set


def get_docs(cluster_id, hac_model, cluster_doc):
    """ Get the ids of all the documents that belong to a cluster.

    Args:
        cluster_id (int): The id of the cluster.
        hac_model (int): The hierarchical model that describes the data.
        cluster_doc (pandas Series): Matches clusters to their documents.
    Returns:
        docs (:list:'int'): A list of the ids of the documents that belong to the
            specified cluster.

    """
    docs = []
    children = hac_model.children_  # The children of each non-leaf node.

    # Move down the tree starting from the cluster with id=cluster_id.
    unvisited_clusters = [cluster_id]
    while len(unvisited_clusters) > 0:
        curr_cluster_id = unvisited_clusters.pop(0)
        if curr_cluster_id < hac_model.n_leaves_:
            docs.extend(cluster_doc[curr_cluster_id])
        else:
            unvisited_clusters.extend(
                children[curr_cluster_id - cluster_doc.size])

    return docs


def display_wordcloud(cluster_id, hac_model, cluster_word):
    """ Display a wordcloud representation of a cluster.

    Args:
        cluster_id (int): The id of the cluster.
        hac_model (int): The hierarchical model that describes the data.
        cluster_word (pandas Series): Matches clusters to their most common words.

    """
    words = []

    children = hac_model.children_  # The children of each non-leaf node.

    # Move down the tree starting from the cluster with id=cluster_id.
    unvisited_clusters = [cluster_id]
    while len(unvisited_clusters) > 0:
        curr_cluster_id = unvisited_clusters.pop(0)
        if curr_cluster_id < hac_model.n_leaves_:
            words.extend(cluster_word[curr_cluster_id])
        else:
            unvisited_clusters.extend(
                children[curr_cluster_id - cluster_word.size])

    wordcloud = WordCloud().generate(text=' '.join(words))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def get_word_rep(cluster_id, hac_model, cluster_word):
    """ Get a three word representation of a cluster.

    Args:
        cluster_id (int): The id of the cluster.
        hac_model (int): The hierarchical model that describes the data.
        cluster_word (pandas Series): Matches clusters to their most common words.

    """
    words = []

    children = hac_model.children_  # The children of each non-leaf node.

    # Move down the tree starting from the cluster with id=cluster_id.
    unvisited_clusters = [cluster_id]
    while len(unvisited_clusters) > 0:
        curr_cluster_id = unvisited_clusters.pop(0)
        if curr_cluster_id < hac_model.n_leaves_:
            words.extend(cluster_word[curr_cluster_id][:3])
        else:
            unvisited_clusters.extend(
                children[curr_cluster_id - cluster_word.size])
    return words[:3]


def browse(corpus, hac_model, n_clusters, cluster_doc, cluster_word):
    """ Simulates Scatter/Gather browsing.

    Args:
        hac_model (:obj:'sklearn.cluster.AgglomerativeClustering'): A hieararchical model
            of a document collection.
        n_clusters (int): The number of clusters presented to the user after each scatter step.
        cluster_doc (:obj:'pandas.Series'): A matching between cluster ids and the ids of their
            documents.

    """
    root_id = hac_model.n_leaves_ + len(hac_model.children_) - 1
    # Generate the inital cluster working set.
    working_set = [root_id]
    curr_dir = os.getcwd()
    if os.path.exists(curr_dir + '/simulation'):
        shutil.rmtree(
            curr_dir + '/simulation')  # Overwrite directory if it exists.
    os.makedirs(curr_dir + '/simulation')

    for i in range(n_clusters):

        start_time = time.time()
        working_set = scatter(hac_model, working_set, n_clusters)
        print('Iteration: ' + str(i + 1) + ' - ' +
              str(round((time.time() - start_time) / 60)) + "' " +
              str(round((time.time() - start_time) % 60)) + "''")
        word_reps = [get_word_rep(cluster, hac_model, cluster_word)
                     for cluster in working_set]
        print('Scatter: ')
        for i, cluster in enumerate(working_set):
            print('%s: (%s, %s, %s)' %
                  (i, word_reps[i][0], word_reps[i][1], word_reps[i][2]))

        gather_finished = False
        command_regex = re.compile(r'%([^\s]+)(\s.+)')
        while not gather_finished:
            input_str = input('--> ')
            res = command_regex.search(input_str)
            if res:
                command = res.group(1)
                print(command)
            else:
                print('Invalid command.')
                continue

            if command == 'show':
                display_wordcloud(working_set[int(res.group(2).strip())],
                                  hac_model, cluster_word)
            elif command == 'sel':
                selected_clusters = [int(x)
                                     for x in res.group(2).strip().split(',')]
                # Check if clusters exist in the working set.
                for cluster_ind in selected_clusters:
                    if cluster_ind not in range(len(working_set)):
                        print('Unknown cluster  %d' % cluster)
                        gather_finished = False
                        break
                    else:
                        gather_finished = True

        working_set = [working_set[i] for i in selected_clusters]
        print('Gather: ' + str(working_set))

        # Log iteration result by saving each cluster's document titles.
        iter_dir_path = curr_dir + '/simulation/' + str(i + 1)
        if os.path.exists(iter_dir_path):
            shutil.rmtree(iter_dir_path)
        os.makedirs(iter_dir_path)

        if sorted(working_set).pop() < hac_model.n_leaves_:
            answer = input('Save results? (yes, no)')
            if answer == 'yes':
                for cluster_id in working_set:
                    cluster_filepath = iter_dir_path + '/' + str(cluster_id)
                    with open(cluster_filepath, 'w+') as cluster_file:
                        doc_titles = [
                            corpus.get_title(docid)
                            for docid in get_docs(cluster_id,
                                                  hac_model, cluster_doc)
                        ]
                        cluster_file.write('\n'.join(doc_titles))
            break  # Only leaves remain in the working set.

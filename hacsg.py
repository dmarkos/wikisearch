# -*- coding: utf-8 -*-
""" Performs a simulated Scatter/Gather browsing and extracts execution information."""

import time
import math


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

        if curr_cluster_id < hac_model.n_leaves_: # Only leaves remain in the working set.
            return working_set

        working_set.pop()
        working_set.extend(children[curr_cluster_id
                                    - hac_model.n_leaves_]) # First 100 ids represent leaves.
        #  print([len(get_docs(cluster_id, hac_model, cluster_doc)) for cluster_id in working_set])

    return working_set

def gather(working_set):
    """ Simulates a user selecting half of the clusters each time.

    Args:
        working_set (:list:'int'): The ids of the clusters in the working set.
    Returns:
        new_working_set (:list:'int'): The ids of the selected clusters.

    """
    new_working_set = working_set[:math.floor(len(working_set)/2)]

    return new_working_set

def get_docs(cluster_id, hac_model, cluster_doc):
    """ Get the ids of all the documents that belong to a cluster.

    Args:
        cluster_id (int): The id of the cluster.
    Returns:
        docs (:list:'int'): A list of the ids of the documents that belong to the
            specified cluster.

    """
    docs = []
    children = hac_model.children_ # The children of each non-leaf node.

    # Move down the tree starting from the cluster with id=cluster_id.
    unvisited_clusters = [cluster_id]
    while len(unvisited_clusters) > 0:
        curr_cluster_id = unvisited_clusters.pop(0)
        if curr_cluster_id < hac_model.n_leaves_:
            docs.extend(cluster_doc[curr_cluster_id])
        else:
            unvisited_clusters.extend(children[curr_cluster_id - 100])

    return docs

def browse(hac_model, n_clusters):
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
    working_set = scatter(hac_model, [root_id], n_clusters)

    for i in range(5):
        start_time = time.time()
        print('Scatter: ' + str(working_set))
        working_set = gather(working_set)
        print('Gather: ' + str(working_set))
        working_set = scatter(hac_model, working_set, n_clusters)
        print('Iteration: ' + str(i+1) + ' - ' + str(round((time.time()-start_time)/60)) + "' "
              + str(round((time.time()-start_time)%60)) + "''")

        if sorted(working_set).pop() < hac_model.n_leaves_:
            break # Only leaves remain in the working set.


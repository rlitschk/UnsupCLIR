import faiss
import numpy as np
from timer import Timer as PPrintTimer


def get_idx(result_query):
    results, query = result_query
    # get index of query in result list
    tmp = np.where(results == query)
    # return index as int or None if query not in results
    if tmp[0].shape[0] == 0:
        return None
    else:
        return tmp[0].tolist()[0]


def search(index, vector_arry, queries=None):
    """
    Iterative execution of queries on a faiss index. This is required for cases where a non-iterative appraoch would be
    too memory-hungry as in experiment_europarl. For CLEF experiments the faiss index can directly answer all queries
    at once.
    :param index: faiss index
    :param vector_arry: all queries
    :param queries: select subset
    :return:
    """
    timer = PPrintTimer()
    nd = len(vector_arry)
    knn = int(0.1 * nd) if nd > 100000 else nd

    if queries is None:
        queries = np.array(range(len(vector_arry)))

    multiplier = 2
    remaining_queries = queries
    all_eval = []
    while remaining_queries.size > 0:
        print("%s queries left %s" % (str(remaining_queries.size), timer.pprint_lap()))
        _, result_list = index.search(vector_arry[remaining_queries], knn)
        tmp_eval = np.array(list(map(get_idx, zip(result_list, remaining_queries.reshape([remaining_queries.size, 1])))))

        where_results_found = tmp_eval is not None
        all_eval = np.concatenate((all_eval, tmp_eval[where_results_found]))
        remaining_queries = remaining_queries[not where_results_found]
        knn *= multiplier
    return all_eval


# if you modify this function, be careful with this:
# https://github.com/facebookresearch/faiss/issues/45
def create_index(vectors, dim=300):
    nlist = 5
    nprobe = nlist
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = nprobe
    return index, quantizer

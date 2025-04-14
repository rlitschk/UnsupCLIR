from itertools import compress

import faiss
import numpy as np

from evaluate import evaluate_clef
from text2vec import create_text_representations, text2vec_idf_sum
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


def run_biencoder_experiment(aggregation_method, query_lang, doc_lang, experiment_data, timer, initialized_embeddings, 
                             tokenize_fn, processes=40):
    """
    Constructs text representations for queries and documents according to the specified aggregation method. From the
    text representations it retrieves for each query the documents and computes the evaluation metric.
    :param aggregation_method:
    :param query_lang:
    :param doc_lang:
    :param experiment_data:
    :param initialized_embeddings:
    :param processes:
    :param timer:
    :return:
    """
    # unpacking values
    qlang_short, qlang_long = query_lang
    dlang_short, dlang_long = doc_lang
    doc_ids, documents, query_ids, queries, relass = experiment_data
    embeddings = initialized_embeddings

    doc_arry = create_text_representations(language=dlang_long, id_text=zip(doc_ids, documents),
                                           emb=embeddings, processes=processes, method=aggregation_method,
                                           idf_weighing=aggregation_method == text2vec_idf_sum,
                                           tokenize_fn=tokenize_fn)
    query_arry = create_text_representations(language=qlang_long, id_text=zip(query_ids, queries),
                                             emb=embeddings, processes=processes, method=aggregation_method,
                                             # Queries are not idf-scaled
                                             idf_weighing=False,
                                             tokenize_fn=tokenize_fn)
    print("Query- and Document-Embeddings created %s" % (timer.pprint_lap()))

    # keep only documents for which we have a non-zero text embedding, i.e. for which at least one
    # word embedding could exists (filters out empty documents)
    doc_non_zero = np.all(doc_arry != 0, axis=1)
    doc_arry = doc_arry[doc_non_zero]
    doc_ids = list(compress(doc_ids, doc_non_zero))
    
    index, quantizer = create_index(doc_arry)
    D, I = index.search(query_arry, len(doc_arry))
    print("Retrieval done %s" % (timer.pprint_lap()))

    # import ir_measures 
    # from ir_measures import AP
    # result = ir_measures.calc_aggregate([AP], run={str(query_ids[i]): {doc_ids[did]: score for did, score in zip(I[i].tolist(), D[i].tolist())} for i in range(len(query_ids))}, qrels={str(qid): {did: 1 for did in relass[qid]} for qid in relass})

    all_rankings, evaluation_result = evaluate_clef(query_ids=query_ids, doc_ids=doc_ids, relass=relass, all_rankings=I)
    return all_rankings, evaluation_result

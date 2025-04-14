import numpy as np


def evaluate_clef(query_ids, doc_ids, relass, all_rankings):
    """
    Evaluates results for queries in terms of Mean Average Precision (MAP). Evaluation gold standard is
    loaded from the relevance assessments.
    :param query_ids: internal id of query
    :param doc_ids: internal id of document
    :param relass: gold standard (expected) rankings
    :param all_rankings: (actual) rankings retrieved
    :return:
    """
    average_precision = []
    qid2ap = {}
    rankings_with_doc_ids = []
    for j in range(len(query_ids)):
        query_id = query_ids[j]  
        if query_id in relass:  
            relevant_docs = set(relass[query_id])
            
            ranking = all_rankings[j].tolist()  # get ranking for j'th query
            ranking_with_doc_ids = [doc_ids[i] for i in ranking]
            rankings_with_doc_ids.append((query_id, ranking_with_doc_ids))
            
            is_relevant = [ranked_doc in relevant_docs for ranked_doc in ranking_with_doc_ids]
            ranks_of_relevant_docs = np.where(is_relevant)[0].tolist()
            precisions = []
            for k, rank in enumerate(ranks_of_relevant_docs, 1):
                summand = k / (rank + 1)  # +1 because of mismatch btw. one based rank and zero based indexing
                precisions.append(summand)
            
            # Skip empty queries, e.g. qid 79 in en-de only consists of single-character string "." 
            # (only applies to Europarl). Out of 1k queries, this applies to less than 5 queries
            if not precisions:
                print(f"Skipping empty query: {query_id}")
                continue
            else:
                ap = np.mean(precisions)
            qid2ap[query_id] = ap
            average_precision.append(ap)
    
    # print(qid2ap)
    mean_average_precision = np.mean(np.array(average_precision))
    return rankings_with_doc_ids, mean_average_precision

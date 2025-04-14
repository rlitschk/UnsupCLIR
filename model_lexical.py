import unicodedata
import numpy as np
import constants as c
from evaluate import evaluate_clef
from model_dense import create_index
from text2vec import clean, tokenize, lookup

from collections import Counter
from itertools import repeat, chain, compress
from functools import partial
from multiprocessing.pool import Pool



def _count_words(document):
    tokens = document
    # {k: v/n_d for k, v in Counter(tokens).items()}
    return dict(Counter(tokens))


def _score_doc_unigram_lm(data, mu=1000):
    doc_id, document_distribution, query_distr, collection_dist, dc = data
    
    n_d = sum(document_distribution.values()) 
    
    if n_d == 0:
        return 0

    smoothing_term = n_d / (n_d + mu)
    document_score = 0

    for query_term, occurrences in query_distr.items():
        if query_term in collection_dist:
            query_freq_in_doc = document_distribution.get(query_term, 0)
            P_q_d = query_freq_in_doc / n_d

            query_freq_in_collection = collection_dist.get(query_term, 0)
            assert query_freq_in_collection != 0
            P_q_dc = query_freq_in_collection / dc

            score = smoothing_term * P_q_d + (1 - smoothing_term) * P_q_dc
            document_score += (np.log(score) * occurrences)

    # calculations up to here were done in log-space
    document_score = np.exp(document_score) if document_score != 0 else 0
    return document_score


def run_unigram_lm(query_lang, doc_lang, experiment_data, timer, processes=40, most_common=None, tokenize_fn=None):
    """
    Builds a unigram language model
    :param query_lang:
    :param doc_lang:
    :param experiment_data:
    :param processes:
    :param most_common:
    :param timer:
    :return:
    """
    _, qlang_long = query_lang
    _, dlang_long = doc_lang
    doc_ids, documents, query_ids, queries, relass = experiment_data
    pool = Pool(processes=processes)

    print("Start preprocessing data %s" % timer.pprint_lap())
    clean_to_lower = partial(clean, to_lower=True)
    tokenize_doc_language = partial(tokenize, language=dlang_long, exclude_digits=True, tokenize_fn=tokenize_fn)
    documents = pool.map(clean_to_lower, documents)
    documents = pool.map(tokenize_doc_language, documents)
    print("Documents preprocessed %s" % (timer.pprint_lap()))
    
    tokenize_query_language = partial(tokenize, language=qlang_long, exclude_digits=True, tokenize_fn=tokenize_fn)
    queries = pool.map(clean_to_lower, queries)
    queries = pool.map(tokenize_query_language, queries)
    print("queries preprocessed %s" % timer.pprint_lap())

    # word frequency distribution per document
    document_distributions = pool.map(_count_words, documents)
    print("Document conditional counts collected %s" % timer.pprint_lap())

    # word frequency distribution per query
    query_distributions = pool.map(_count_words, queries)
    print("Query conditional counts collected %s" % timer.pprint_lap())

    collection_size = sum([sum(document.values()) for document in document_distributions])
    collection_distribution = Counter()
    for document in document_distributions:
        collection_distribution.update(document)  # { token: frequency }
    if most_common is not None:
        collection_distribution.most_common(most_common)
    collection_distribution = dict(collection_distribution)
    print("Marginal counts collected %s" % timer.pprint_lap())

    np.random.seed(10)
    random_ranking = np.random.permutation(len(documents))
    doc_count = len(document_distributions)
    broadcasted_collection_size = [collection_size] * doc_count

    results = []
    print("start evaluation %s" % timer.pprint_lap())
    for i, query in enumerate(query_distributions, 1):
        query_id = query_ids[i - 1]
        suffix = ""
        if query_id in relass:
            lm_uni = partial(_score_doc_unigram_lm)
            scores_for_query = pool.map(lm_uni, zip(doc_ids, document_distributions,  # {word_d: freq}
                                                                   repeat(query),  # {word_q: freq}
                                                                   repeat(collection_distribution),
                                                                   broadcasted_collection_size))  # {word_dc: freq}
            # condition for random ranking if all documents score zero
            any_score_non_zero = sum(scores_for_query) > 0
            # sort by argsort if we have non-zero scores, otherwise use random ranking
            ranking_for_query = np.argsort(-np.array(scores_for_query)) if any_score_non_zero else random_ranking
            results.append(ranking_for_query)
        else:
            results.append(random_ranking)  # query without relevant documents is not fired
            suffix = " --> no relevant docs for q_id %s" % str(query_id)
        if i % 10 == 0:
            print("%s  queries processed (%s) %s" % (i, timer.pprint_lap(), suffix))

    pool.close()
    pool.join()
    
    all_rankings, evaluation_result = evaluate_clef(query_ids=query_ids, doc_ids=doc_ids, relass=relass,
                                                    all_rankings=np.array(results))
    return all_rankings, evaluation_result


def run_wordbyword_translation(query_lang, doc_lang, experiment_data, initialized_embeddings, timer, tokenize_fn, 
                               processes=40):
    # unpacking values
    qlang_short, qlang_long = query_lang
    dlang_short, dlang_long = doc_lang
    doc_ids, documents, query_ids, queries, relass = experiment_data
    embeddings = initialized_embeddings

    queries = list(map(clean, queries))
    tokenize_language = partial(tokenize, language=qlang_long, tokenize_fn=tokenize_fn)
    queries_tokenized = list(map(tokenize_language, queries))
    unique_query_terms = list(set(chain(*queries_tokenized)))

    doc_language_vocabulary = [k for k, v in embeddings.lang_vocabularies[dlang_long].items()]
    doc_language_embeddings = embeddings.lang_embeddings[dlang_long]

    print("Vocabulary extraced %s" % timer.pprint_lap())

    index, quantizer = create_index(doc_language_embeddings)
    search_vecs = []
    zero_vec = np.zeros(c.EMBEDDING_DIM, dtype=np.float32)
    keep_words_as_is = set()
    for unique_query_term in unique_query_terms:
        lookedup_word, vec = lookup(unique_query_term, qlang_long, embedding_lookup=embeddings)
        if vec is not None:
            search_vecs.append(vec)
        else:
            keep_words_as_is.add(lookedup_word)
            search_vecs.append(zero_vec)

    search_vecs = np.array(search_vecs, dtype=np.float32)
    non_zeros = np.all(search_vecs != 0, axis=1)
    # Word embeddings of english (query language) words
    search_vecs = search_vecs[non_zeros]
    unique_query_terms = list(compress(unique_query_terms, non_zeros))

    nearest_neighbors = 1
    # search in index of (document language) embeddings/words
    _, I = index.search(search_vecs, nearest_neighbors)
    print("Nearest neighbors / translation mapping computed %s" % timer.pprint_lap())

    nearest_neighbors_of_unique_query_terms = [doc_language_vocabulary[nearest_neighbor.tolist()[0]] for
                                               nearest_neighbor in I]
    nearest_neighbor_mapping = dict(zip(unique_query_terms, nearest_neighbors_of_unique_query_terms))

    def translate(query):
        translation = []
        for query_term in query:
            if query_term in keep_words_as_is:
                translation.append(query_term)
                continue

            translated_query_term = None
            if query_term in nearest_neighbor_mapping:
                translated_query_term = nearest_neighbor_mapping[query_term]
            elif query_term.lower() in nearest_neighbor_mapping:
                translated_query_term = nearest_neighbor_mapping[query_term.lower()]

            if translated_query_term is None:
                no_special_chars = ''.join(
                    (c for c in unicodedata.normalize('NFD', query_term) if unicodedata.category(c) != 'Mn'))
                if no_special_chars in nearest_neighbor_mapping:
                    translated_query_term = nearest_neighbor_mapping[no_special_chars]
                elif no_special_chars.lower() in nearest_neighbor_mapping:
                    translated_query_term = nearest_neighbor_mapping[no_special_chars.lower]
            translation.append(translated_query_term)
        return ' '.join([word for word in translation if word is not None])

    translated_queries = list(map(translate, queries_tokenized))
    print("Queries translated, now running Unigram LM %s" % timer.pprint_lap())
    new_experiment_data = doc_ids, documents, query_ids, translated_queries, relass
    return run_unigram_lm(query_lang=query_lang,
                          doc_lang=doc_lang,
                          experiment_data=new_experiment_data,
                          timer=timer,
                          processes=processes,
                          tokenize_fn=tokenize_fn)

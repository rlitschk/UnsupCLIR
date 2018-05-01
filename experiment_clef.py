import os
import os.path
import numpy as np
import unicodedata
from itertools import repeat
from itertools import chain
from functools import partial
from itertools import compress
from collections import Counter
from multiprocessing.pool import Pool
from faiss_index import create_index

import constants as c
from embeddings import Embeddings
from timer import Timer as PPrintTimer
from load_data import load_queries
from load_data import load_clef_documents
from load_data import load_relevance_assessments
from text2vec import lookup
from text2vec import clean
from text2vec import tokenize
from text2vec import text2vec_sum
from text2vec import text2vec_bigram
from text2vec import text2vec_idf_sum
from text2vec import create_text_representations
from collection_extractors import extract_dutch
from collection_extractors import extract_italian_sda9495
from collection_extractors import extract_italian_lastampa
from collection_extractors import extract_finish_aamuleth9495


timer = PPrintTimer()


def _word_overlap(document, query):
    """
    Take items from document and keep if they are in the query, i.e. return intersection of query and document words
    and take the frequency from the document.
    :param document:
    :param query:
    :return:
    """
    document = dict(document)
    query = dict(query)
    return {k: v for k, v in document.items() if k in query.keys()}


def _count_words(document):
    tokens = document
    # {k: v/n_d for k, v in Counter(tokens).items()}
    return dict(Counter(tokens))


def _score_doc_unigram_lm(data, mu=1000):
    document_distribution, query_distr, collection_dist, dc = data
    n_d = sum(query_distr.values())  # document length

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


def _save_ranking(config_str, all_rankings, base_path):
    """
    Stores ranking in rankings_year_langpair_embspace_aggrMethod.txt file, which is later reused for computing
    ensembled rankings of different aggregations methods.
    :param config_str: csv record string
    :param all_rankings: ranking to be stored
    :param base_path: directory where file should be saved
    :return:
    """
    _, campaign_year, language_pair, embedding_space, aggregation_method, _, _ = config_str.split(";")
    subdir = "rankings_%s_%s_%s_%s.txt" % (campaign_year, language_pair, embedding_space, aggregation_method)
    path = base_path + subdir
    file_content = []
    for query, ranking in all_rankings:
        one_line = str(query) + '; ' + ' '.join(ranking) + "\n"
        file_content.append(one_line)
    file_content = ''.join(file_content)
    with open(path, mode="w") as ranking_file:
        ranking_file.write(file_content)
    pass


def prepare_experiment(doc_dirs, limit_documents, query_file, limit_queries,
                       query_language, relevance_assessment_file):
    """
    Loads documents, evaluation data and queries needed to run different experiments on CLEF data.
    :param doc_dirs: directories containing the corpora for a specific CLEF campaign
    :param limit_documents: for debugging purposes -> limit number of docs loaded
    :param query_file: CLEF Topics (i.e., query) file
    :param limit_queries: for debugging purposes -> limit number of queries loaded
    :param query_language: language of queries
    :param relevance_assessment_file: relevance assesment file
    :return:
    """
    if limit_documents is not None:
        limit_documents -= 1
    documents = []
    doc_ids = []
    limit_reached = False
    for doc_dir, extractor in doc_dirs:
        if not limit_reached:
            for file in next(os.walk(doc_dir))[2]:
                tmp_doc_ids, tmp_documents = load_clef_documents(doc_dir + file, extractor, limit_documents)
                documents.extend(tmp_documents)
                doc_ids.extend(tmp_doc_ids)
                if len(documents) == limit_documents:
                    limit_reached = True
                    break
    print("Documents loaded %s" % (timer.pprint_lap()))
    relass = load_relevance_assessments(relevance_assessment_file)
    print("Evaluation data loaded %s" % (timer.pprint_lap()))
    query_ids, queries = load_queries(query_file, language_tag=query_language, limit=limit_queries)
    print("Queries loaded %s" % (timer.pprint_lap()))
    return doc_ids, documents, query_ids, queries, relass


def prepare_word_embeddings(query_lang_emb, qlang_long,
                            doc_lang_emb, dlang_long,
                            limit_emb, normalize=False, processes=40):
    """
    Creates Word Embedding Helper Object
    :param query_lang_emb: language of queries
    :param qlang_long: short version
    :param doc_lang_emb: language of documents
    :param dlang_long: short version
    :param limit_emb: load only first n embeddings
    :param normalize: transform to unit vectors
    :param processes: number of parallel workers
    :return:
    """
    embeddings = Embeddings()
    embeddings.load_embeddings(query_lang_emb, processes=processes, language=qlang_long,
                               limit=limit_emb, normalize=normalize)
    embeddings.load_embeddings(doc_lang_emb, processes=processes, language=dlang_long,
                               limit=limit_emb, normalize=normalize)
    return embeddings


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
    rankings_with_doc_ids = []
    for j in range(len(query_ids)):
        query_id = query_ids[j]  # for the ith query
        if query_id in relass:  # len(relevant_docs) > 0:
            relevant_docs = relass[query_id]
            ranking = all_rankings[j].tolist()  # get ranking for j'th query

            ranking_with_doc_ids = [doc_ids[i] for i in ranking]
            rankings_with_doc_ids.append((query_id, ranking_with_doc_ids))

            is_relevant = [ranked_doc in relevant_docs for ranked_doc in ranking_with_doc_ids]
            ranks_of_relevant_docs = np.where(is_relevant)[0].tolist()
            precisions = []
            for k, rank in enumerate(ranks_of_relevant_docs, 1):
                summand = k / (rank + 1)  # +1 because of mismatch btw. one based rank and zero based indexing
                precisions.append(summand)
            average_precision.append(np.mean(precisions))
    mean_average_precision = np.mean(np.array(average_precision))
    return rankings_with_doc_ids, mean_average_precision


def run_unigram_lm(query_lang, doc_lang, experiment_data, processes=40, most_common=None):
    """
    Builds a unigram language model
    :param query_lang:
    :param doc_lang:
    :param experiment_data:
    :param processes:
    :param most_common:
    :return:
    """
    _, qlang_long = query_lang
    _, dlang_long = doc_lang
    doc_ids, documents, query_ids, queries, relass = experiment_data
    pool = Pool(processes=processes)

    print("Start preprocessing data %s" % timer.pprint_lap())
    clean_to_lower = partial(clean, to_lower=True)
    tokenize_doc_language = partial(tokenize, language=dlang_long, exclude_digits=True)
    documents = pool.map(clean_to_lower, documents)
    documents = pool.map(tokenize_doc_language, documents)
    print("Documents preprocessed %s" % (timer.pprint_lap()))

    tokenize_query_language = partial(tokenize, language=qlang_long, exclude_digits=True)
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
            doc_subset_distributions = pool.starmap(_word_overlap, zip(document_distributions, repeat(query)))
            col_subset_distribution = _word_overlap(collection_distribution, query)

            scores_for_query = pool.map(_score_doc_unigram_lm, zip(doc_subset_distributions,  # {word_d: freq}
                                                                   repeat(query),  # {word_q: freq}
                                                                   repeat(col_subset_distribution),
                                                                   # collection_dist_subsets,
                                                                   broadcasted_collection_size))  # {word_dc: freq}
            # condition for random ranking if all documents score zero
            any_score_non_zero = sum(scores_for_query) > 0
            # https://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
            # ranking_for_query = np.lexsort((-np.array(scores_for_query),random_ranking))
            # sort first by argsort then by random ranking
            ranking_for_query = np.argsort(-np.array(scores_for_query)) if any_score_non_zero else random_ranking
            results.append(ranking_for_query)
        else:
            results.append(random_ranking)  # query without relevant documents is not fired
            suffix = " --> no relevant docs for q_id %s" % str(query_id)
        print("%s  queries processed (%s) %s" % (i, timer.pprint_lap(), suffix))

    pool.close()
    pool.join()
    all_rankings, evaluation_result = evaluate_clef(query_ids=query_ids, doc_ids=doc_ids, relass=relass,
                                                    all_rankings=np.array(results))
    return all_rankings, evaluation_result


def run_wordbyword_translation(query_lang, doc_lang, experiment_data, initialized_embeddings, processes=40):
    # unpacking values
    qlang_short, qlang_long = query_lang
    dlang_short, dlang_long = doc_lang
    doc_ids, documents, query_ids, queries, relass = experiment_data
    embeddings = initialized_embeddings

    queries = list(map(clean, queries))
    tokenize_language = partial(tokenize, language=qlang_long)
    queries_tokenized = list(map(tokenize_language, queries))
    unique_query_terms = list(set(chain(*queries_tokenized)))

    doc_language_vocabulary = [k for k, v in embeddings.lang_vocabularies[dlang_long].items()]
    doc_language_embeddings = embeddings.lang_embeddings[dlang_long]

    print("Vocabulary extraced %s" % timer.pprint_lap())

    index, quantizer = create_index(doc_language_embeddings)
    search_vecs = []
    zero_vec = np.zeros(300, dtype=np.float32)
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
                          processes=processes)


def run_experiment(aggregation_method, query_lang, doc_lang, experiment_data, initialized_embeddings, processes=40):
    """
    Constructs text representations for queries and documents according to the specified aggregation method. From the
    text representations it retrieves for each query the documents and computes the evaluation metric.
    :param aggregation_method:
    :param query_lang:
    :param doc_lang:
    :param experiment_data:
    :param initialized_embeddings:
    :param processes:
    :return:
    """
    # unpacking values
    qlang_short, qlang_long = query_lang
    dlang_short, dlang_long = doc_lang
    doc_ids, documents, query_ids, queries, relass = experiment_data
    embeddings = initialized_embeddings

    doc_arry = create_text_representations(language=dlang_long, id_text=zip(doc_ids, documents),
                                           emb=embeddings, processes=processes, method=aggregation_method,
                                           idf_weighing=aggregation_method == text2vec_idf_sum)
    query_arry = create_text_representations(language=qlang_long, id_text=zip(query_ids, queries),
                                             emb=embeddings, processes=processes, method=aggregation_method,
                                             idf_weighing=False)  # Queries are not idf-scaled
    print("Query- and Document-Embeddings created %s" % (timer.pprint_lap()))

    # keep only documents for which we have a non-zero text embedding, i.e. for which at least one
    # word embedding could exists (filters out empty documents)
    doc_non_zero = np.all(doc_arry != 0, axis=1)
    doc_arry = doc_arry[doc_non_zero]
    doc_ids = list(compress(doc_ids, doc_non_zero))

    index, quantizer = create_index(doc_arry)
    D, I = index.search(query_arry, len(doc_arry))
    print("Retrieval done %s" % (timer.pprint_lap()))

    all_rankings, evaluation_result = evaluate_clef(query_ids=query_ids, doc_ids=doc_ids, relass=relass, all_rankings=I)
    return all_rankings, evaluation_result


def run(experiment, name, vspace, experiment_count, offset, results, csv_prefix):
    """
    Executes configured experiments and records result in csv record
    :param experiment: configured experiment
    :param name: name of the aggregation method to be used
    :param vspace: name of the method used for inducing shared embedding space
    :param experiment_count: used for resuming experiments
    :param offset: value of experiment_count from where it should be resumed
    :param results: containing caching result csv records
    :param csv_prefix: string that is prefixed to each csv record
    :return:
    """
    if not experiment_count <= offset:
        tmp_timer = PPrintTimer().start()  # Experiment timer
        rankings, eval_score = experiment()
        time = tmp_timer.pprint_stop(suffix=False)
        result = "%s;%s;%s;%s;%s\n" % (str(experiment_count), csv_prefix + vspace, name, time, str(eval_score))
        _save_ranking(result, rankings, c.RESULTS_DIR)
        results.append(result)
        print("\n" + result)
    return experiment_count + 1, results


def main():
    process_count = c.PROCESS_COUNT  # number of cores
    query_limit = None  # limit for testing/debugging,e.g. 10
    doc_limit = None  # limit for testing/debugging, e.g. 100
    emb_limit = 100000  # 100,000 is value used in paper
    most_frequent_vocab = None

    # Prepare dutch CLEF data
    nl_all = (c.PATH_BASE_DOCUMENTS + "dutch/all/", extract_dutch)
    dutch = {"2001": [nl_all], "2002": [nl_all], "2003": [nl_all]}

    # Prepare italian CLEF data
    it_lastampa = (c.PATH_BASE_DOCUMENTS + "italian/la_stampa/", extract_italian_lastampa)
    it_sda94 = (c.PATH_BASE_DOCUMENTS + "italian/sda_italian_94/", extract_italian_sda9495)
    it_sda95 = (c.PATH_BASE_DOCUMENTS + "italian/sda_italian_95/", extract_italian_sda9495)
    italian = {"2001": [it_lastampa, it_sda94],
               "2002": [it_lastampa, it_sda94],
               "2003": [it_lastampa, it_sda94, it_sda95]}

    # Prepare finnish CLEF data
    aamu9495 = c.PATH_BASE_DOCUMENTS + "finnish/aamu/"
    fi_ammulethi9495 = (aamu9495, extract_finish_aamuleth9495)
    finnish = {"2001": None, "2002": [fi_ammulethi9495], "2003": [fi_ammulethi9495]}
    _all = {"dutch": dutch, "italian": italian, "finnish": finnish}

    # Offset experiment counter and continue from where it has stopped before
    if os.path.exists(c.RESULTS_FILE):
        with open(c.RESULTS_FILE, mode="r") as f:
            next(f)  # Skip header line
            for line in f:
                offset = int(line.split(";")[0])
    else:
        offset = -1

    counter = 0  # zero-based counter of experiments conducted
    # mean_AP = 0 # mean_AP variable exist only for debugging purposes
    # rankings = [] # empty list exist only for debugging purposes
    csv_records = []  # gets serialized into results file later

    if offset == -1:
        csv_records = ["Counter;Year;LangPair;EmbSpaceMethod;AggrMethod;duration;MAP\n"]

    for english, target_language in c.LANGUAGE_PAIRS:
        for year in c.YEARs:
            lang_pair = "%s%s" % (english[0], target_language[0])
            csv_prefix = year + ";" + lang_pair + ";"

            # The CLEF 2001 campaign does not include finnish
            if year == "2001" and target_language[0] == 'fi':
                continue

            current_assessment_file = c.PATH_BASE_EVAL + year + "/qrels_" + target_language[1] + "_" + year
            current_path_documents = _all[target_language[1]][year]
            current_path_queries = c.PATH_BASE_QUERIES + year + "/Top-en" + year[-2:] + ".txt"

            current_experiment_data = prepare_experiment(doc_dirs=current_path_documents,
                                                         limit_documents=doc_limit,
                                                         query_file=current_path_queries,
                                                         limit_queries=query_limit,
                                                         query_language='en',
                                                         relevance_assessment_file=current_assessment_file)
            print("Experiment data for %s (%s) loaded %s" % (lang_pair, year, timer.pprint_lap()))

            def experiment():
                return run_unigram_lm(query_lang=english,
                                      doc_lang=target_language,
                                      experiment_data=current_experiment_data,
                                      processes=process_count,
                                      most_common=most_frequent_vocab)
            counter, csv_records = run(experiment, "UnigramLM", "None", counter, offset, csv_records, csv_prefix)

            for vector_space in c.METHODs:
                path_prefix = c.PATH_EMB_BASE + vector_space + "/" + lang_pair
                current_query_embeddings_file = path_prefix + "/en.vectors"
                current_doc_embeddings_file = path_prefix + "/" + target_language[0] + ".vectors"

                if not (counter + 3) <= offset:
                    # loading embeddings takes time, load only if they're not offset/left unused
                    word_embeddings = prepare_word_embeddings(query_lang_emb=current_query_embeddings_file,
                                                              qlang_long=english[1],
                                                              doc_lang_emb=current_doc_embeddings_file,
                                                              dlang_long=target_language[1],
                                                              limit_emb=emb_limit,
                                                              normalize=True)
                    print("Word embeddings for %s loaded %s" % (vector_space, timer.pprint_lap()))
                    # Prepare all further runs
                    run_configured_experiment = partial(run_experiment,
                                                        query_lang=english,
                                                        doc_lang=target_language,
                                                        experiment_data=current_experiment_data,
                                                        processes=process_count,
                                                        initialized_embeddings=word_embeddings)

                # Term by Term Translation with Unigram Language Model, termed TbT-QT in paper
                def experiment():
                    return run_wordbyword_translation(query_lang=english,
                                                      doc_lang=target_language,
                                                      experiment_data=current_experiment_data,
                                                      processes=process_count,
                                                      initialized_embeddings=word_embeddings)
                counter, csv_records = run(experiment, "TbTQT", vector_space, counter, offset, csv_records, csv_prefix)

                # Sum word embeddings, termed BWE-Agg-Add in paper
                def experiment():
                    return run_configured_experiment(aggregation_method=text2vec_sum)
                counter, csv_records = run(experiment, "Sum", vector_space, counter, offset, csv_records, csv_prefix)

                # Sum of idf-weighted word embeddings (idf for documents only), termed BWE-Agg-IDF in paper
                def experiment():
                    return run_configured_experiment(aggregation_method=text2vec_idf_sum)
                counter, csv_records = run(experiment, "IDFSum", vector_space, counter, offset, csv_records, csv_prefix)

                # bigram aggregation method by blunsom and hermann (http://www.aclweb.org/anthology/P14-1006)
                def experiment():
                    return run_configured_experiment(aggregation_method=text2vec_bigram)
                counter, csv_records = run(experiment, "Bigram", vector_space, counter, offset, csv_records, csv_prefix)

            # Updates results.csv after all vector-spaces for a single lanugage pair and a single year have been run
            duration = timer.pprint_lap()
            print("Year %s, Language-Pair %s done! (%s)" % (year, lang_pair, duration))
            if len(csv_records) > 0:
                with open(c.RESULTS_FILE, mode="a") as f:
                    for line in csv_records:
                        f.write(line)
                csv_records = []
    pass


if __name__ == "__main__":
    main()
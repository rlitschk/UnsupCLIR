from timer import Timer as PPrintTimer
import numpy as np
import random
import constants as c
from multiprocessing import Pool
from load_data import load_relevance_assessments
from functools import partial


# year, language_pair, embeddingspace are fixed, methods are variable
def run_ensemble(method_paths, relass, param_model_weights=None):
    """
    Returns for a set of specified aggregation methods the (weighted) averaged ranking. I.e., each document receives a
    new ensemble score defined as the average rank of all provided methods. And the new ranking results from the
    re-ranking of those average ranks.
    :param method_paths: pairs of tuples t_i=(aggregation_method, path_to_rankings), aggregation_method is e.g. IDFSum
    :param relass: relevance assessment file
    :param param_model_weights: ensemble weight for each aggregation_method
    :return:
    """
    all_documents = set()
    method_count = len(method_paths)

    # if no weighting specified assign equal weights
    if param_model_weights is None:
        model_weights = [1.0 / method_count for _ in range(method_count)]
    else:
        model_weights = param_model_weights

    def load_rankings(path):
        """
        Reads file for of results from one specific aggregation method and vector space induction method and
        turns ranking lines "query_id; doc_id doc_id ...\n" into dict.
        :param path: relevance assessment file
        :return: {q_id: {d_id: rank}}
        """
        rankings_current_method = {}
        with open(path, mode="r") as f:
            for line in f:
                tmp_query, documents_str = line.split(";")
                tmp_query = int(tmp_query)
                tmp_ranked_documents = documents_str.strip().split(" ")
                assert len(set(tmp_ranked_documents)) == len(tmp_ranked_documents)
                all_documents.update(tmp_ranked_documents)
                doc_rank = {d_id: r for r, d_id in enumerate(tmp_ranked_documents, 1)}
                rankings_current_method[tmp_query] = doc_rank
        return rankings_current_method

    query_rankings_method_x = [load_rankings(method_path[1]) for method_path in method_paths]
    average_precisions = []

    for query_id, relevant_docs in relass.items():
        if len(relevant_docs) > 0:
            ensemble_scores = []
            excluded_docs = 0
            for doc_id in all_documents:
                new_ranking_score = 0
                for k, method in enumerate(query_rankings_method_x):
                    try:
                        new_ranking_score += (model_weights[k] * method[query_id][doc_id])
                    except KeyError:
                        # If for an embedding-based method there's no sentence vector then document is excluded.
                        # This happens on very rare cases, e.g. for empty documents.
                        new_ranking_score = -1
                        break

                if new_ranking_score == -1:
                    excluded_docs += 1
                    continue
                else:
                    ensemble_scores.append((new_ranking_score, doc_id))

            # sanity check / debugging breakpoint
            if excluded_docs > 10:
                pass

            # if two documents have the same score, shuffle randomly
            ranking_with_doc_ids = sorted(ensemble_scores, key=lambda v: (v[0], random.random()))
            is_relevant = [ranked_doc[1] in relevant_docs for ranked_doc in ranking_with_doc_ids]
            ranks_of_relevant_docs = np.where(is_relevant)[0].tolist()
            precisions = []
            for k, rank in enumerate(ranks_of_relevant_docs, 1):
                summand = k / (rank + 1)  # +1 because of mismatch btw. one based rank and zero based indexing
                precisions.append(summand)
            average_precisions.append(np.mean(precisions))

    mean_average_precision = np.mean(np.array(average_precisions))
    log_str = ';'.join([method for method, path in method_paths]) + ";"
    if param_model_weights is not None:
        log_str += ','.join([str(weight) for weight in model_weights])
    else:
        log_str += "all_equal"
    return log_str, mean_average_precision


def main():
    # template for specifying path to results files, e.g. 'PROJECT_HOME/Results/rankings_2001_enit_Conneau.txt'
    file_template = c.RESULTS_DIR + "rankings_%s_%s_%s_%s.txt"
    process_count = c.PROCESS_COUNT
    # "UnigramLM" can be included manually by uncommenting unigram_configuration lines below
    aggregation_methods = ["TbTQT", "IDFSum"]  # ,"Bigram","Sum"]
    _lambda_combinations = [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)]

    pool = Pool(processes=process_count)
    timer = PPrintTimer()
    all_results = []
    for year in c.YEARs:
        for lang_pair in c.LANGUAGE_PAIRS_SHORT:
            # load relevance assesment files
            if lang_pair.endswith("it"):
                relass_file = c.PATH_BASE_EVAL + year + "/qrels_italian_" + year
            elif lang_pair.endswith("fi"):
                if int(year) == 2001:
                    continue  # 2001 campaign does not include finnish
                relass_file = c.PATH_BASE_EVAL + year + "/qrels_finnish_" + year
            elif lang_pair.endswith("nl"):
                relass_file = c.PATH_BASE_EVAL + year + "/qrels_dutch_" + year
            relevance_assessments = load_relevance_assessments(relass_file)

            # all combinations of vector space induction and aggregation method, e.g.
            # (Conneau_TbTQT, '/path/to/rankings/produced/by/this/method.txt')
            unique_configurations = [("%s_%s" % (vectorspace, aggregation_method),
                                      file_template % (year, lang_pair, vectorspace, aggregation_method))
                                     for vectorspace in c.METHODs
                                     for aggregation_method in aggregation_methods]
            # unigram_configuration = ('None_Unigram-LM', template % (year, lang_pair, 'None', 'Unigram-LM'))
            # unique_configurations.append(unigram_configuration)

            # run each combination with different weighting schemas, ensemble only methods from same shared vector space
            tmp_results = []
            for weight_combination in _lambda_combinations:
                #
                tmp_combinations_of_two = [(unique_configurations[0], unique_configurations[1]),
                                           (unique_configurations[2], unique_configurations[3]),
                                           (unique_configurations[4], unique_configurations[5])]
                run = partial(run_ensemble, relass=relevance_assessments, param_model_weights=weight_combination)
                print("Running %s combinations for %s, %s with weights %s" %
                      (str(len(tmp_combinations_of_two)), lang_pair, year, weight_combination))
                tmp_results.extend(pool.map(run, tmp_combinations_of_two))

            # tmp_combinations_of_three = ...
            # run = partial(run_ensemble, relass=relevance_assessments, param_model_weights=None)
            # print("running %s combinations for %s, %s" % (str(len(tmp_combinations_of_three)), lang_pair, year))
            # tmp_results.extend(pool.map(run, tmp_combinations_of_three))

            all_results.extend(
                ["%s;%s;%s;%s" % (year, lang_pair, method_score[0], method_score[1]) for method_score in tmp_results])
            best = str(max([tmp_result[1] for tmp_result in tmp_results]))
            print("Done! (%s) best in this run: %s" % (timer.pprint_lap(), best))

    with open(c.ENSEMBLE_RESULTS_FILE, mode="x") as f:
        for result in all_results:
            f.write(result + "\n")

    print("Evaluating all ensemble models done! (%s)" % (timer.pprint_stop()))


if __name__ == "__main__":
    main()

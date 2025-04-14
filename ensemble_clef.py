import os.path
import argparse
import tqdm
import random
import constants as c
import numpy as np

from clef_dataloaders import load_relevance_assessments
from timer import Timer as PPrintTimer
from helper import print_summary


parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=str, required=True)
parser.add_argument("--clwe", nargs="+", help=f"(optional) Choose one or more: {c.CLWEs}", choices=c.sigir18_CLWEs, required=False, default=c.sigir18_CLWEs)
parser.add_argument("--years", nargs="+", choices=["2001", "2002", "2003"])
parser.add_argument("--language_pairs", nargs="*", required=True)

args = parser.parse_args()
YEARS = args.years
LANGUAGE_PAIRS = []
for lp in args.language_pairs:
    ql, dl = lp[:2], lp[2:]
    if lp not in ["ennl", "enit", "enfi"]:
        raise RuntimeError(f"sigir18 paper does not contain {lp} evaluation")
    LANGUAGE_PAIRS.append((c.short2pair[ql], c.short2pair[dl]))
CLWE = args.clwe
DIR = args.directory


random.seed(0)

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
    return mean_average_precision


def main():
    print("Start evaluating ensembles...")
    file_template = os.path.join(DIR, "rankings_%s_%s_%s_%s.txt")

    # "UnigramLM" can be included manually by uncommenting unigram_configuration lines below
    aggregation_methods = ["TbTQT", "IDFSum"] 
    _lambda_combinations = [(0.5, 0.5), (0.7, 0.3)]

    timer = PPrintTimer()
    langpair2year2emb_space2model2map = {}
    for qlang, dlang in LANGUAGE_PAIRS:
        print(f"{qlang[0]}->{dlang[0]}")
        year2emb_space2model2map = {}
        for year in c.YEARS:
            if year == "2001" and qlang[0] + dlang[0] == "enfi":
                continue

            relevance_assessments = load_relevance_assessments(language=dlang[0], year=year)

            emb_space2model2map = {}
            for clwe in tqdm.tqdm(CLWE):
                individual_models = [
                    (model, file_template % (year, qlang[0]+dlang[0], clwe, model))
                    for model in aggregation_methods
                ]
                model2map = {}
                for weight_combination in _lambda_combinations:
                    model_1, _ = individual_models[0]
                    model_2, _ = individual_models[1]
                    model = f"ensemble_{model_1}={weight_combination[0]}_{model_2}={weight_combination[1]}"
                    _map = run_ensemble(param_model_weights=weight_combination, method_paths=individual_models, relass=relevance_assessments)
                    model2map[model] = _map
                
                emb_space2model2map[clwe] = model2map
            
            year2emb_space2model2map[year] = emb_space2model2map
        
        langpair2year2emb_space2model2map[(qlang[0], dlang[0])] = year2emb_space2model2map
        # print_summary(langpair2year2emb_space2model2map, LANGUAGE_PAIRS, is_pilot=True)
    
    rows = print_summary(langpair2year2emb_space2model2map, LANGUAGE_PAIRS, is_pilot=True)
    
    with open(os.path.join(DIR, "ensemble_results.csv"), mode="w") as f:
        f.writelines([l + "\n" for l in rows])

    print("Evaluating all ensemble models done! (%s)" % (timer.pprint_stop()))


if __name__ == "__main__":
    main()

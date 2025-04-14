import os
import os.path
import random
import constants as c
import argparse
import numpy as np
from functools import partial
from embeddings import get_embeddings_fn
from helper import print_summary, save_results_csv
from model_lexical import run_unigram_lm, run_wordbyword_translation # , run_unigram_lm_bug
from model_dense import run_biencoder_experiment
from timer import Timer as PPrintTimer
from clef_dataloaders import load_queries
from clef_dataloaders import load_documents
from clef_dataloaders import load_relevance_assessments
from text2vec import text2vec_sum
from text2vec import text2vec_idf_sum
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from load_europarl import *
from constants import short2pair
import pickle


timer = PPrintTimer()

# required
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True, choices=["clef", "europarl"])
parser.add_argument("--is_pilot", action="store_true", help="add --is_pilot to run sigir18 experiments.")

# not required 
parser.add_argument("--clwe", nargs="+", help=f"(optional) Choose one or more: {c.CLWEs}", choices=c.CLWEs, required=False, default=[])
parser.add_argument("--embeddings_dir", type=str, required=False)
parser.add_argument("--models", nargs="+", help=f"(optional) Choose one or more: {c.RUN_MODELS}", choices=c.RUN_MODELS, default=c.RUN_MODELS, required=False)
parser.add_argument("--language_pairs", nargs="*", required=True)
parser.add_argument("--years", nargs="+", choices=["2001", "2002", "2003"], required=False)
parser.add_argument("--path_translated_queries", type=str, required=False)
parser.add_argument("--tokenizer", type=str, choices=["word", "wordpunct"], default="wordpunct")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--save_rankings", type=bool, default=False)

args = parser.parse_args()
MODELS = args.models
CLWE = args.clwe
RESULTS_DIR = args.output_dir
DATASET = args.dataset
YEARS = args.years
PILOT = args.is_pilot
EMBEDDINGS_DIR = args.embeddings_dir
PATH_TRANSLATED_QUERIES = args.path_translated_queries
SAVE_RANKINGS = args.save_rankings

TOKENIZER = word_tokenize if args.tokenizer == "word" else wordpunct_tokenize
SEED=args.seed

print(f"Tokenizer: {TOKENIZER}")
print(f"PILOT={PILOT}")
print(f"SEED={SEED}")
print(f"SAVE_RANKINGS={SAVE_RANKINGS}")

random.seed(SEED)
np.random.seed(SEED)


LANGUAGE_PAIRS = []
for lp in args.language_pairs:
    qlang = lp[:2]
    dlang = lp[2:]
    if lp not in ["ennl", "enit", "enfi"] and PILOT:
        raise RuntimeError(f"sigir18 paper does not contain {lp} evaluation")
    LANGUAGE_PAIRS.append((short2pair[qlang], short2pair[dlang]))


if not PILOT:
    YEARS = ["2003"]


if "MT-IR" in MODELS and PATH_TRANSLATED_QUERIES is None: 
    raise RuntimeError("Path to query translations is missing")


if "MT-IR" in MODELS and ("2002" in YEARS or "2001" in YEARS):
    raise RuntimeError("We only translated CLEF 2003 queries.")


if PATH_TRANSLATED_QUERIES and not os.path.exists(PATH_TRANSLATED_QUERIES):
    raise FileNotFoundError(f"Path to query translations does not exist: {PATH_TRANSLATED_QUERIES}")


if PILOT and DATASET != "clef":
    raise RuntimeError(f"sigir18 paper does not contain {DATASET} evaluation")


emb_limit = 100_000 if PILOT else 200_000
if DATASET == "europarl": assert PILOT is False


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
    eval_score = None
    if not experiment_count <= offset:
        tmp_timer = PPrintTimer().start()  # Experiment timer
        rankings, eval_score = experiment()
        time = tmp_timer.pprint_stop(suffix=False)
        result = "%s;%s;%s;%s;%s\n" % (str(experiment_count), csv_prefix + vspace, name, time, str(eval_score))
        print("\n" + result)
        if SAVE_RANKINGS:
            _save_ranking(result, rankings, RESULTS_DIR)
        results.append(result)
    return experiment_count + 1, results, eval_score


def get_clef_data(source_language, target_language, year):
    query_ids, queries = load_queries(source_language[0], year) # , limit=qlimit)
    doc_ids, documents = load_documents(target_language[0], year)
    relass = load_relevance_assessments(target_language[0], year)

    if "MT-IR" in MODELS:
        with open(os.path.join(PATH_TRANSLATED_QUERIES, f"{source_language[0]}_translated_to_{target_language[0]}.pickle"), "rb") as f:
            translated_queries = pickle.load(f)
    else:
        translated_queries = None

    experiment_data = doc_ids, documents, query_ids, queries, relass
    return experiment_data, translated_queries


def get_europarl_data(source_language, target_language, _):
    # last argument is not used and only included to match interface of get_clef_data()
    src = source_language[0]
    tgt = target_language[0]

    src_eval_file = f"europarl/{src}-{tgt}/Europarl.{src}-{tgt}.{src}.1k.queries"
    tar_eval_file = f"europarl/{src}-{tgt}/Europarl.{src}-{tgt}.{tgt}.100k.documents"

    queries = load_txt_data(src_eval_file)#, limit=limit_examples)
    documents = load_txt_data(tar_eval_file)#, limit=limit_examples)

    queries, documents, skipped_src_rows = clean_and_rm_duplicates(queries, documents) # remove duplicates

    if "MT-IR" in MODELS:
        src_eval_file = os.path.join(PATH_TRANSLATED_QUERIES, f"{src}-{tgt}/Europarl.{src}-{tgt}.{src}.1k.queries.translated")
        translated_queries = load_txt_data(src_eval_file)#, limit=limit_examples)
        translated_queries = [clean(q) for qid, q in enumerate(translated_queries) if qid not in skipped_src_rows]
    else:
        translated_queries = None

    query_ids = list(range(len(queries))) # zip(range(len(queries)), queries)
    doc_ids = [str(_id) for _id in range(len(documents))] # zip(range(len(documents)), documents)

    relass = {qid: [str(qid)] for qid in query_ids}
    experiment_data = doc_ids, documents, query_ids, queries, relass
    return experiment_data, translated_queries


def main():
    process_count = c.PROCESS_COUNT  # number of cores
    unigram_lm_most_frequent_vocab = None
    counter = 0  # zero-based counter of experiments conducted
    offset = -1
    get_data = get_clef_data if DATASET == "clef" else get_europarl_data

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"results dir = {RESULTS_DIR}")

    # gets serialized into results file later
    csv_records = ["Counter;Year;LangPair;EmbSpaceMethod;AggrMethod;duration;MAP\n"]

    langpair2year2emb_space2model2map = {}
    for source_language, target_language in LANGUAGE_PAIRS:
        year2emb_space2model2map = {}
        for year in YEARS:
            lang_pair = "%s%s" % (source_language[0], target_language[0])
            csv_prefix = year + ";" + lang_pair + ";"

            # The CLEF 2001 campaign does not include finnish
            skip_finnish_2001 = year == "2001" and target_language[0] == 'fi'

            # europarl does not contain russian
            skip_russian_europarl = DATASET == "europarl" and target_language[0] == "ru"

            if skip_finnish_2001 or skip_russian_europarl:
                continue

            current_experiment_data, translated_queries = get_data(source_language, target_language, year)
            print("Experiment data for %s (%s) loaded %s" % (lang_pair, year, timer.pprint_lap()))
            emb_space2model2map = {}
            model2map = {}

            #
            # Unigram LM
            # 
            if "LM-UNI" in MODELS:
                def experiment():
                    return run_unigram_lm(query_lang=source_language,
                                          doc_lang=target_language,
                                          experiment_data=current_experiment_data,
                                          timer=timer,
                                          processes=process_count,
                                          most_common=unigram_lm_most_frequent_vocab,
                                          tokenize_fn=TOKENIZER)
                counter, csv_records, mean_ap = run(experiment, "UnigramLM", "None", counter, offset, csv_records, csv_prefix)
                if mean_ap: model2map["UnigramLM"] = mean_ap

            #
            # Query Translation (Queries pre-translated with Google Translate)
            # 
            if "MT-IR" in MODELS and not PILOT:
                doc_ids, documents, query_ids, _, relass = current_experiment_data
                translated_experiment_data = doc_ids, documents, query_ids, translated_queries, relass

                def experiment():
                    return run_unigram_lm(query_lang=source_language,
                                          doc_lang=target_language,
                                          experiment_data=translated_experiment_data,
                                          timer=timer,
                                          processes=process_count,
                                          most_common=unigram_lm_most_frequent_vocab,
                                          tokenize_fn=TOKENIZER)
                counter, csv_records, mean_ap = run(experiment, "MT-IR", "None", counter, offset, csv_records, csv_prefix)
                if mean_ap: model2map["MT-IR"] = mean_ap

            if model2map:
                emb_space2model2map["None"] = model2map

            for vector_space in CLWE:
                model2map = {}
                word_embeddings_fn = get_embeddings_fn(
                    emb_limit, source_language, target_language, vector_space, PILOT, EMBEDDINGS_DIR
                )

                # we do NOT normalize individual CLWEs BoW-Agg-Add or BoW-Agg-IDF 
                # (cosine similarity length normalizes queries and documents)
                word_embeddings = word_embeddings_fn(normalize=False)
                print("Word embeddings for %s loaded %s" % (vector_space, timer.pprint_lap()))

                # Prepare all further runs
                run_configured_experiment = partial(run_biencoder_experiment,
                                                    query_lang=source_language,
                                                    doc_lang=target_language,
                                                    experiment_data=current_experiment_data,
                                                    timer=timer,
                                                    processes=process_count,
                                                    initialized_embeddings=word_embeddings,
                                                    tokenize_fn=TOKENIZER)

                #
                # TbT-QT (Term by Term Translation with Unigram Language Model)
                #
                if "TbT-QT" in MODELS:
                    def experiment():
                        return run_wordbyword_translation(query_lang=source_language,
                                                          doc_lang=target_language,
                                                          experiment_data=current_experiment_data,
                                                          timer=timer,
                                                          processes=process_count,
                                                          # pre-normalize individual CLWEs for TbT-QT (L2 normalization) 
                                                          initialized_embeddings=word_embeddings_fn(normalize=True),
                                                          tokenize_fn=TOKENIZER)
                    counter, csv_records, mean_ap = run(experiment, "TbTQT", vector_space, counter, offset, csv_records, csv_prefix)
                    if mean_ap: model2map["TbTQT"] = mean_ap

                #
                # BoW-Agg-Add
                #
                if DATASET == "clef" and "SUM" in MODELS: # and PILOT:
                    def experiment():
                        return run_configured_experiment(aggregation_method=text2vec_sum)
                    counter, csv_records, mean_ap = run(experiment, "Sum", vector_space, counter, offset, csv_records, csv_prefix)
                    if mean_ap: model2map["Sum"] = mean_ap

                #
                # BoW-Agg-IDF (idf for documents only)
                # 
                if "IDF-SUM" in MODELS:
                    def experiment():
                        return run_configured_experiment(aggregation_method=text2vec_idf_sum)
                    counter, csv_records, mean_ap = run(experiment, "IDFSum", vector_space, counter, offset, csv_records, csv_prefix)
                    if mean_ap: model2map["IDFSum"] = mean_ap

                emb_space2model2map[vector_space] = model2map
            year2emb_space2model2map[year] = emb_space2model2map

            # Updates results.csv after all vector-spaces for a single lanugage pair and a single year have been run
            duration = timer.pprint_lap()
            print("Year %s, Language-Pair %s done! (%s)" % (year, lang_pair, duration))

        langpair2year2emb_space2model2map[(source_language[0], target_language[0])] = year2emb_space2model2map
        print_summary(langpair2year2emb_space2model2map, LANGUAGE_PAIRS, PILOT)

    rows = print_summary(langpair2year2emb_space2model2map, LANGUAGE_PAIRS, PILOT)
    save_results_csv(csv_records, rows, RESULTS_DIR)
    print(TOKENIZER)
    print("all done!")


if __name__ == "__main__":
    main()

import numpy as np
import constants as c
from itertools import compress
from embeddings import Embeddings
from timer import Timer as PPrintTimer
from load_data import load_txt_data
from text2vec import clean
from text2vec import create_text_representations
from faiss_index import create_index
from faiss_index import search


def clean_mt_data(src, tar):
    src_unique = set()
    tar_unique = set()
    src_clean = []
    tar_clean = []

    for s, t in zip(src, tar):
        skip_record = False
        if s not in src_unique:
            src_unique.add(s)
        else:
            skip_record = True
        if t not in tar_unique:
            tar_unique.add(t)
        else:
            skip_record = True
        if not skip_record:
            src_clean.append(clean(s))
            tar_clean.append(clean(t))
    return src_clean, tar_clean


def run_experiment(src_lang, src_file, targ_lan, targ_file, eval_files,
                   limit_emb=500000, limit_examples=None, normalize=False,
                   two_way=True):
    timer = PPrintTimer()

    # load and share word embeddings
    embeddings = Embeddings()
    embeddings.load_embeddings(src_file, processes=40, language=src_lang, limit=limit_emb, normalize=normalize)
    embeddings.load_embeddings(targ_file, processes=40, language=targ_lan, limit=limit_emb, normalize=normalize)
    print("Embeddings loaded %s" % (timer.pprint_lap()))

    # load machine translation data
    src_eval_file, tar_eval_file = eval_files
    src_lines = load_txt_data(src_eval_file, limit=limit_examples)
    tar_lines = load_txt_data(tar_eval_file, limit=limit_examples)

    # remove duplicates
    src_lines, tar_lines = clean_mt_data(src_lines, tar_lines)

    # sentence embeddings
    src_arry = create_text_representations(language=src_lang, id_text=src_lines, emb=embeddings)
    tar_arry = create_text_representations(language=targ_lan, id_text=tar_lines, emb=embeddings)
    print("MT data loaded %s" % (timer.pprint_lap()))

    # remove sentences where no word embedding could be retrieved in the source language
    src_non_zero = np.all(src_arry != 0, axis=1)
    src_arry = src_arry[src_non_zero]
    tar_arry = tar_arry[src_non_zero]
    src_lines = list(compress(src_lines, src_non_zero))
    tar_lines = list(compress(tar_lines, src_non_zero))

    # remove sentences where no word embedding could be retrieved in the target language
    tar_non_zero = np.all(tar_arry != 0, axis=1)
    src_arry = src_arry[tar_non_zero]
    tar_arry = tar_arry[tar_non_zero]
    src_lines = list(compress(src_lines, tar_non_zero))
    tar_lines = list(compress(tar_lines, tar_non_zero))

    # create index
    index, q = create_index(tar_arry)
    print("Index built %s" % (timer.pprint_lap()))

    # sample queries
    nd = len(src_lines)
    nq = 5000
    np.random.seed(0)
    queries = np.random.choice(nd, nq, replace=False)

    # query index src_lan -> tar_lan
    all_eval = search(index, src_arry, queries)
    print("%s -> %s Search done %s" % (src_lang, targ_lan, timer.pprint_lap()))

    mrr = np.mean(1 / (all_eval + 1))  # mean reciprocal rank
    harmonic_mean = np.power(mrr, -1)  # harmonic mean of ranks
    print(str(mrr) + "-> MRR")
    print(str(harmonic_mean) + "-> (harmonic) mean of ranks")

    # query index tar_lan -> src_lan
    if two_way:
        index, q = create_index(src_arry)
        all_eval = search(index, tar_arry, queries)
        print("%s -> %s Search done %s" % (targ_lan, src_lang, timer.pprint_lap()))

        mrr = np.mean(1 / (all_eval + 1))
        harmonic_mean = np.power(mrr, -1)
        print(str(mrr) + "-> MRR")
        print(str(harmonic_mean) + "-> (harmonic) mean of ranks")

    print("total duration: %s" % timer.pprint_stop())
    pass


run_experiment('german', c.PATH_EMB_DE, 'english', c.PATH_EMB_EN, (c.PATH_MT_DE, c.PATH_MT_EN), normalize=False)


# def inspect_example(q_id, top_k, result_lists, distances):
#     query = queries[q_id].tolist()
#     print("\nQuery:\n (id: %s) %s" % (query, src_lines[query][0:50]))
#     print("Expected:\n (id: %s) (%s) %s " % (query, np.dot(src_arry[query], tar_arry[query]),
#                                              tar_lines[query][0:50]))
#     print("Actual:")
#     for k in range(top_k):
#         print("(id: %s)(%s) %s" % (result_lists[q_id][k], distances[q_id][k],
#                                    tar_lines[result_lists[q_id][k]][0:50]))

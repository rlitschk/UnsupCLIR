import os.path
import pickle
import numpy as np

from functools import partial
from constants import EMBEDDING_DIM
from multiprocessing import Pool


class Embeddings(object):
    """Captures functionality to load and store textual embeddings"""

    def __init__(self):
        self.lang_embeddings = {}
        self.lang_emb_norms = {}
        self.lang_vocabularies = {}
        self.emb_sizes = {}

    def get_vector(self, lang, word):
        if word in self.lang_vocabularies[lang]:
            return self.lang_embeddings[lang][self.lang_vocabularies[lang][word]]
        else:
            return None

    def set_vector(self, lang, word, vector):
        if word in self.lang_vocabularies[lang]:
            self.lang_embeddings[lang][self.lang_vocabularies[lang][word]] = vector

    def add_language(self, lang):
        self.lang_vocabularies[lang] = {}
        self.lang_emb_norms[lang] = []
        self.lang_embeddings[lang] = []
        self.lang_emb_norms[lang] = []

    def load_embeddings(self, filepath, language='en', processes=1, limit=None, normalize=False):
        with open(filepath) as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line)
                if i == limit:
                    break

        pool = Pool(processes=processes)
        emb = pool.map(_load_embedding, lines)  # entry[0] = word, entry[1] = vector, entry[2] = norm
        pool.close()
        pool.join()
        emb = [entry for entry in emb if entry[1] is not None and entry[1].shape[0] == EMBEDDING_DIM]  # cleaning

        self.add_language(language)
        index_correction = 0
        for i, entry in enumerate(emb):
            if entry[0] in self.lang_vocabularies[language]:
                #print("Warning: duplicate embeddings for %s" % entry[0])
                index_correction += 1  # keep index integrity
                continue
            term = entry[0]
            term_embedding = entry[1]
            term_emb_norm = entry[2]
            self.lang_vocabularies[language][term] = (i - index_correction)

            if normalize:
                self.lang_embeddings[language].append(term_embedding / term_emb_norm)
            else:
                self.lang_embeddings[language].append(term_embedding)

            self.lang_emb_norms[language].append(term_emb_norm)

        self.lang_embeddings[language] = np.array(self.lang_embeddings[language], dtype=np.float32)
        self.emb_sizes[language] = self.lang_embeddings[language].shape[1]

    def load_embeddings_from_memory(self, vocabulary, embs, language):
        self.lang_embeddings[language] = np.vstack(embs)
        self.lang_emb_norms[language] = self.lang_embeddings[language]
        self.emb_sizes[language] = self.lang_embeddings[language].shape[1]
        self.lang_vocabularies[language] = {word: index for (index, word) in enumerate(vocabulary)}


def _load_embedding(line):
    if line != "\n":
        splt = line.split()
        word = ''.join(splt[:-EMBEDDING_DIM])
        embedding = np.array(splt[-EMBEDDING_DIM:], dtype=np.float32)
        norm = np.linalg.norm(embedding, 2)
        return word, embedding, norm
    else:
        return None, None, None


def get_embeddings_fn(emb_limit, source_language, target_language, vector_space, is_pilot, embeddings_path):
    lang_pair = "%s%s" % (source_language[0], target_language[0])
    if is_pilot:
        # path_prefix = c.PATH_EMB_BASE + vector_space + "/" + lang_pair
        path_prefix = os.path.join(embeddings_path, vector_space, lang_pair)
        current_query_embeddings_file = path_prefix + "/en.vectors"
        current_doc_embeddings_file = path_prefix + "/" + target_language[0] + ".vectors"
        word_embeddings_fn = partial(
            prepare_word_embeddings,
            query_lang_emb=current_query_embeddings_file,
            qlang_long=source_language[1],
            doc_lang_emb=current_doc_embeddings_file,
            dlang_long=target_language[1],
            limit_emb=emb_limit,
        )
    else:
        src = source_language[0]
        tgt = target_language[0]

        clwe2emb_path = {
            "icp": "ft.wiki.%s.300.vectors",
            "cca": "vectors_%s-%s.%s.yacle.train.freq.5k.np",
            "muse": "fasttext.%s-%s.unsup.%s.vec.embeddings",
            "proc": "vectors_%s-%s.%s.yacle.train.freq.5k.np",
            "procb": "%s-%s.%s.yacle.train.freq.1k.vectors",
            "rcsls": "fasttext.%s-%s.yacle.train.freq.5k.%s.vec.embeddings",
            "vecmap": "fasttext.%s-%s.unsup.%s.vec.embeddings"
        }
        clwe2vocab_path = {
            "icp": "ft.wiki.%s.300.vocab",
            "cca": "vocab_%s-%s.%s.yacle.train.freq.5k.pkl",
            "muse": "fasttext.%s-%s.unsup.%s.vec.vocabulary",
            "proc": "vocab_%s-%s.%s.yacle.train.freq.5k.pkl",
            "procb": "%s-%s.%s.yacle.train.freq.1k.vocab",
            "rcsls": "fasttext.%s-%s.yacle.train.freq.5k.%s.vec.vocabulary",
            "vecmap": "fasttext.%s-%s.unsup.%s.vec.vocabulary"
        }

        # path = os.path.join(c.PATH_EMB_BASE, f"{vector_space}/{src}-{tgt}/")
        path = os.path.join(embeddings_path, f"{vector_space}/{src}-{tgt}/")
        if vector_space == "icp":
            query_lang_emb = os.path.join(path, "proj_src.vectors.npy")
            query_lang_vocab = os.path.join(path, clwe2vocab_path[vector_space] % src)
            doc_lang_emb = os.path.join(path, clwe2emb_path[vector_space] % tgt)
            doc_lang_vocab = os.path.join(path, clwe2vocab_path[vector_space] % tgt)
        else:
            doc_lang_emb = os.path.join(path, clwe2emb_path[vector_space] % (src, tgt, tgt))
            doc_lang_vocab = os.path.join(path, clwe2vocab_path[vector_space] % (src, tgt, tgt))
            query_lang_emb = os.path.join(path, clwe2emb_path[vector_space] % (src, tgt, src))
            query_lang_vocab = os.path.join(path, clwe2vocab_path[vector_space] % (src, tgt, src))

        word_embeddings_fn = partial(
            prepare_word_embeddingsv2,
            # prepare_word_embeddings,
            query_lang_emb=query_lang_emb,
            query_lang_vocab=query_lang_vocab,
            qlang_long=source_language[1],
            doc_lang_emb=doc_lang_emb,
            doc_lang_vocab=doc_lang_vocab,
            dlang_long=target_language[1],
            limit_emb=emb_limit
        )
    return word_embeddings_fn


def prepare_word_embeddingsv2(query_lang_emb, query_lang_vocab, qlang_long,
                              doc_lang_emb, doc_lang_vocab, dlang_long,
                              limit_emb, normalize):
    embeddings = Embeddings()
    with open(query_lang_emb, "rb") as f:
        src_emb = np.load(f)
    src_emb = src_emb[:limit_emb]
    with open(query_lang_vocab, "rb") as f:
        src_vocab = pickle.load(f)
    src_vocab = {term: term_id for term, term_id in src_vocab.items() if term_id < limit_emb}
    
    if normalize:
        src_emb = src_emb/np.expand_dims(np.linalg.norm(src_emb, axis=1),1)
    embeddings.load_embeddings_from_memory(vocabulary=src_vocab, embs=src_emb, language=qlang_long)

    with open(doc_lang_emb, "rb") as f:
        tgt_emb = np.load(f)
    tgt_emb = tgt_emb[:limit_emb]
    with open(doc_lang_vocab, "rb") as f:
        tgt_vocab = pickle.load(f)
    tgt_vocab = {term: term_id for term, term_id in tgt_vocab.items() if term_id < limit_emb}
    
    if normalize:
        tgt_emb = tgt_emb/np.expand_dims(np.linalg.norm(tgt_emb, axis=1),1)
    embeddings.load_embeddings_from_memory(vocabulary=tgt_vocab, embs=tgt_emb, language=dlang_long)

    return embeddings


def prepare_word_embeddings(query_lang_emb, qlang_long,
                            doc_lang_emb, dlang_long,
                            limit_emb, normalize, processes=40):
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

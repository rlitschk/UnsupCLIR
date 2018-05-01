import numpy as np
from multiprocessing import Pool


class Embeddings(object):
    """Captures functionality to load and store textual embeddings"""

    def __init__(self, cache_similarities=False):
        self.lang_embeddings = {}
        self.lang_emb_norms = {}
        self.lang_vocabularies = {}
        self.emb_sizes = {}
        self.cache = {}
        self.do_cache = cache_similarities

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
        emb = [entry for entry in emb if entry[1] is not None and entry[1].shape[0] == 300]  # cleaning

        self.add_language(language)
        index_correction = 0
        for i, entry in enumerate(emb):
            if entry[0] in self.lang_vocabularies[language]:
                print("Warning: duplicate embeddings for %s" % entry[0])
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
        word = splt[0]
        embedding = np.array(splt[1:], dtype=np.float32)
        norm = np.linalg.norm(embedding, 2)
        return word, embedding, norm
    else:
        return None, None, None

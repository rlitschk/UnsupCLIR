import copy
import re
import string
import numpy as np
import unicodedata
import nltk
# from nltk.tokenize import WhitespaceTokenizer as Tokenizer
# # from nltk.tokenize import WordPunctTokenizer as Tokenizer
# from nltk.tokenize.moses import MosesTokenizer as Tokenizer
# from experiment_clef import lookup
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
from functools import partial
from itertools import chain
from collections import Counter


regex = re.compile('[%s]' % re.escape(string.punctuation))


def clean(_str, to_lower=True):
    """
    Cleans string from newlines and punctuation characters
    :param _str:
    :param to_lower:
    :return:
    """
    if to_lower:
        _str = _str.lower()
        # _str = _str.replace("find reports on"," ")
        # _str = _str.replace("find documents", " ")

    if _str is not None:
        _str = _str.replace("\n", " ").replace("\r", " ")
        return regex.sub(' ', _str)
    return None


def tokenize(text, language, exclude_digits=False):
    """
    Call first clean then this function.
    :param exclude_digits: whether include or exclude digits
    :param text: string to be tokenized
    :param language: language flag for retrieving stop words
    :return:
    """
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words(language))
    punctuation = set(string.punctuation)
    tokens = []
    for token in word_tokenize(text, language=language):
        if token not in stopwords and token.lower() not in stopwords and token not in punctuation and len(token) > 1:
            if exclude_digits:
                if not any(t.isdigit() for t in token):
                    tokens.append(token)
            else:
                tokens.append(token)
    return tokens


def lookup(token, language, embedding_lookup=None):
    """
    First try to lookup embedding from word as it appears in text, then its lowercase
    version, then its version with accents stripped, then its its version with accents
    stripped and lowercased letters.
    :param token: term to be looked up
    :param language: language in which term occurs
    :param embedding_lookup: embedding helper used for lookup
    :return:
    """
    local_lookup = embedding_lookup if embedding_lookup is not None else embeddings
    word_vector = local_lookup.get_vector(language, token)
    word = token
    if word_vector is None:
        word_vector = local_lookup.get_vector(language, token.lower())
        word = token.lower()
    if word_vector is None:
        no_special_chars = ''.join((c for c in unicodedata.normalize('NFD', token) if unicodedata.category(c) != 'Mn'))
        word_vector = local_lookup.get_vector(language, no_special_chars)
        word = no_special_chars
    if word_vector is None:
        word_vector = local_lookup.get_vector(language, no_special_chars.lower())
        word = no_special_chars.lower()
    return word, word_vector


def text2vec_bigram(text, language='', dim=300):
    """
    Document aggregation method presented in http://www.aclweb.org/anthology/P14-1006
    :param text:
    :param language:
    :param dim:
    :return:
    """
    _id, txt = text
    text = clean(txt)
    zero_vec = np.zeros(dim)
    tokens = tokenize(text, language=language)

    bigram_vectors = []
    for word_1, word_2 in nltk.bigrams(tokens):
        _, emb_1 = lookup(word_1, language)
        if emb_1 is None:
            emb_1 = zero_vec
        _, emb_2 = lookup(word_2, language)
        if emb_2 is None:
            emb_2 = zero_vec
        bigram_vectors.append(np.tanh(emb_1 + emb_2))

    document_vector = np.sum(bigram_vectors , 0) if len(bigram_vectors) > 0 else zero_vec
    if not np.array_equal(document_vector, zero_vec):
        try:
                document_vector /= np.linalg.norm(document_vector, 2)
        except Exception as e:
            document_vector = zero_vec
    return document_vector, [], [], 0


def text2vec_idf_sum(text, language='', dim=300):
    """
    This function is used used only as a symbol (for a unified interface). The idf-scaling
    occurs in the caller: create_text_representations
    :param text: query / document
    :param language: language of text
    :param dim: dimensionality of word embeddings
    :return:
    """
    return text2vec_sum(text, language=language, dim=dim)


def text2vec_sum(text, language='', dim=300):
    """
    Transforms text into vector representation by embedding lookup on shared/bilingual embedding
    space. The text is represented as a sum of the word embeddings
    :param language: language of text
    :param dim: dimensionality of word embeddings
    :param text: query / document
    :return: unit vector of the sum
    """
    _id, txt = text
    text = clean(txt)

    unknown_words = []
    word_vectors = []
    zero_vec = np.zeros(dim)

    _all=0
    _unique=set()
    for token in tokenize(text, language=language):
        _all+=1
        _unique.add(token)

        _, word_vector = lookup(token, language)
        if word_vector is None: #tv-wereld, oost-duitsland (split and lookup each token)
            word_vector = zero_vec
            if not any(t.isdigit() for t in token):
                unknown_words.append(token)
        elif not word_vector[0] > 500:
            pass
        word_vectors.append(word_vector)

    document_vector = np.sum(word_vectors, 0)
    if not np.array_equal(document_vector, zero_vec):
        try:
            document_vector /= np.linalg.norm(document_vector, 2)
        except:
            document_vector = zero_vec
    return document_vector, unknown_words, _all, len(_unique)


def _map_func(id_text, language):
    """
    Helper function for computing IDF weights, text -> list of words
    :param id_text: (document id, text) tuple
    :param language: text language
    :return:
    """
    _id, doc = id_text
    doc = tokenize(clean(doc), language=language)
    # result = list(set(doc))

    result = []
    for unnormalized_word in list(set(doc)):
        normalized_word, _  = lookup(unnormalized_word,language)
        result.append(normalized_word)

    return result


def compute_idf_weights(text, language, processes):
    """
    Returns a mapping { term: IDF_term }
    :param text: list of documents in corpus
    :param language: corpus language
    :param processes: paralellization parameter
    :return:
    """
    pool = Pool(processes=processes)
    _map_func_language = partial(_map_func, language=language)
    # each occurrence of a word results from one document
    words = list(pool.map(_map_func_language, text))
    pool.close()
    pool.join()

    collection_size = len(text)
    flat_words = list(chain(*words))

    doc_frequencies = dict(Counter(flat_words))
    idf_mapping = {term: np.log(collection_size / doc_frequency) for term, doc_frequency in doc_frequencies.items()}
    return idf_mapping


def create_text_representations(language, id_text, emb, method = text2vec_sum, processes = 40, idf_weighing = False):
    """
    Runs a text2vec method in parallel to transform documents to document vectors. It
    requires a global embedding variable to be created first.
    :param language: used for the look-up table "embeddings"
    :param id_text: (doc_id, document_tokens)
    :param method: textvec variant
    :param emb: embedding
    :param processes: number processes to run in parallel
    :param idf_weighing: whether to rescale word embeddings with the words idf
    :return:
    """
    global embeddings
    embeddings = emb

    id_text = list(id_text)
    if idf_weighing:
        # We modify the embedding object and want to reuse the original one for subsequent experiments
        embeddings = copy.deepcopy(emb)
        # compute IDF weights
        idf_weights = compute_idf_weights(id_text, language, processes)
        print("idf weights computed")

        for key in embeddings.lang_vocabularies[language].keys():
            try:
                idf = idf_weights[key]
                old_embedding = embeddings.get_vector(lang=language,word=key)
                rescaled_embedding = old_embedding * idf
                embeddings.set_vector(lang=language,word=key,vector=rescaled_embedding)
            except:
                pass
        print("old weights idf-rescaled")

    pool = Pool(processes = processes)
    partial_method = partial(method, language=language)
    results = pool.map(partial_method, id_text)
    pool.close()
    pool.join()
    return np.array([result[0] for result in results], dtype=np.float32)

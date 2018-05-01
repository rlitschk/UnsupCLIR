import codecs
from lxml import html as etree


def _decode_xml(path):
    with codecs.open(path, encoding='ISO-8859-1') as f:
        xml = f.read()
        xml = xml.replace("<BODY>", "<BODYY>").replace("</BODY>", "</BODYY>")
        xml = xml.replace("<HEAD>", "<HEADD>").replace("</HEAD>", "</HEADD>")

    xml = "<root>" + xml + "</root>"
    return etree.fromstring(xml)


def load_txt_data(path, limit=None):
    with open(path) as f:
        text = []
        for lcount, line in enumerate(f):
            text.append(line)
            if lcount == limit:
                break
    return text


def load_relevance_assessments(path):
    positive_list = {}
    with open(path) as f:
        for line in f.readlines():
            tokens = line.rstrip("\n").split(" ")
            # check if document is relevant for query
            if int(tokens[len(tokens) - 1]) != 0:
                query_id = int(tokens[0].strip())
                document_id = tokens[2].strip()
                if query_id not in positive_list:
                    relevant_docs = [document_id]
                else:
                    relevant_docs = positive_list[query_id]
                    relevant_docs.append(document_id)
                positive_list[query_id] = relevant_docs
    return positive_list


def load_clef_documents(path, extractor, limit=None):
    tree = _decode_xml(path)
    documents = []
    ids = []

    for i, doc in enumerate(list(tree)):
        if len(documents) == limit:
            break
        document_id, full_text = extractor(doc)
        ids.append(document_id)
        documents.append(full_text)

    return ids, documents


def load_queries(path, language_tag, limit=None):
    tag_title = language_tag + '-title'
    tag_desc = language_tag + '-desc'
    # tag_narr = language_tag + '-narr'
    tree = _decode_xml(path)
    queries = []
    ids = []

    for i, topic in enumerate(list(tree)):
        _id = topic.findtext('num').strip() # e.g. 'C041'
        _id = int(_id[1:]) # e.g. 41
        title = topic.findtext(tag_title)
        desc = topic.findtext(tag_desc)
        query = ' '.join([title, desc])
        # queries.append(clean(query))
        queries.append(query)
        ids.append(_id)
        if i == limit:
            break
    return ids, queries

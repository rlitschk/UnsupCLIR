# from lxml import html as etree


def _find_all_and_concatenate(doc, xpath):
    elements = [paragraph.text for paragraph in doc.findall(xpath) if paragraph.text is not None]
    elements = ' '.join(elements) if elements is not None else " "
    return elements


def _combine(text_elements):
    """
    Removes newlines and carriage returns
    :param text_elements: list of text snippets
    :return: cleaned and concatenated version of it
    """
    full_text = ' '.join(text_elements)
    return full_text.replace("\n", " ").replace("\r", " ")


def extract_dutch(doc):
    document_id = doc.findtext("docid")
    document_title = _find_all_and_concatenate(doc, "bodyy/ti/p")
    lead = _find_all_and_concatenate(doc, "bodyy/le/p")
    text = _find_all_and_concatenate(doc, "bodyy/te/p")
    caption = _find_all_and_concatenate(doc, "bodyy/os/p")
    full_text = _combine([lead, text, document_title, caption])
    return document_id, full_text


def extract_italian_lastampa(doc):
    document_id = doc.findtext("docid")
    document_title = _find_all_and_concatenate(doc, "title")
    text = _find_all_and_concatenate(doc, "text")
    full_text = _combine([text, document_title])
    return document_id, full_text


def extract_italian_sda9495(doc):
    document_id = doc.findtext("docid")
    title = _find_all_and_concatenate(doc, "ti")
    lead = _find_all_and_concatenate(doc, "ld")
    text = _find_all_and_concatenate(doc, "tx")
    full_text = _combine([title, lead, text])
    return document_id, full_text


def extract_finish_aamuleth9495(doc):
    document_id = doc.findtext("docid")
    text = _find_all_and_concatenate(doc, "text")
    text = _combine([text])
    return document_id, text

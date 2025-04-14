import codecs


def load_txt_data(path, limit=None):
    # with open(path) as f:
    with codecs.open(path, encoding="utf8", errors="replace") as f:
        text = []
        for lcount, line in enumerate(f):
            text.append(line)
            if lcount == limit:
                break
    return text


def clean_and_rm_duplicates(src, tar):
    src_unique = set()
    tar_unique = set()
    src_clean = []
    tar_clean = []
    
    # keep track of duplicate queries that are filtered, 
    # so we can filter the same translated queries
    skipped_src_rows = set()
    
    # debugging
    skipped_tgt_rows = set()

    for i, (s, t) in enumerate(zip(src, tar)):
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
        else:
            skipped_src_rows.add(i)
            skipped_tgt_rows.add(i)
    
    for i, t in enumerate(tar[i+1:], start=i+1):
        if t not in tar_unique:
            tar_unique.add(t)
            tar_clean.append(clean(t))
        else:
            skipped_tgt_rows.add(i)
    
    return src_clean, tar_clean, skipped_src_rows


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
        return _str 
    return None


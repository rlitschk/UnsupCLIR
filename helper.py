import os

from datetime import datetime

import constants as c


def print_summary(langpair2year2emb_space2model2map, language_pairs, is_pilot):
    SEP = ";"
    rows = []

    langpairs = [(src[0], tgt[0]) for src, tgt in language_pairs if (src[0], tgt[0]) in langpair2year2emb_space2model2map]
    years = [y for y in c.YEARS if y in langpair2year2emb_space2model2map[langpairs[0]]]
    clwes = [m for m in c.CLWEs if m in langpair2year2emb_space2model2map[langpairs[0]][years[0]]] 
    models = list(langpair2year2emb_space2model2map[langpairs[0]][years[0]][clwes[0]].keys())

    # add header
    header = SEP.join(["CLWE", "Model"] if is_pilot else ["Model", "CLWE"]) + SEP
    cols = []
    for src, tgt in langpairs:
        for year in years:
            col = f"{src+tgt}_{year[-2:]}"
            if not col == "enfi_01": 
                cols.append(col) # enfi didn't exist in 2001
    header += SEP.join(cols)
    rows.append(header)

    # add baselines 
    if "None" in langpair2year2emb_space2model2map[langpairs[0]][years[0]]:
        model2map = langpair2year2emb_space2model2map[langpairs[0]][years[0]]["None"]
        for baseline in ["UnigramLM", "MT-IR"]:
            if baseline in model2map:
                row = "--" + SEP + baseline + SEP
                for src, tgt in langpairs:
                    row += SEP.join([
                        str(langpair2year2emb_space2model2map[(src, tgt)][year]["None"][baseline])
                        for year in years if
                        not (year == "2001" and (src + tgt) == "enfi")
                    ]) + SEP # enfi didn't exist in 2001
                rows.append(row)
    
    # add clwe results
    if is_pilot:
        sort_key = {"Sum": 1, "IDFSum": 2, "TbTQT": 3}
        for clwe in clwes: 
            for model in sorted(models, key=lambda e: sort_key.get(e, 1000)):
                row = clwe + SEP + model + SEP
                for src, tgt in langpairs:
                    row += SEP.join([
                        str(langpair2year2emb_space2model2map[(src, tgt)][year][clwe][model]) for year in years 
                        if not (year == "2001" and src + tgt == "enfi") 
                        and model in langpair2year2emb_space2model2map[(src, tgt)][year][clwe]
                    ]) + SEP
                rows.append(row) 
    else:
        for model in sorted(models): 
            for clwe in clwes:
                row = model + SEP + clwe + SEP
                for src, tgt in langpairs:
                    row += SEP.join([
                        str(langpair2year2emb_space2model2map[(src, tgt)][year][clwe][model]) for year in years 
                        if not (year == "2001" and src + tgt == "enfi") 
                        and model in langpair2year2emb_space2model2map[(src, tgt)][year][clwe]
                    ]) + SEP
                rows.append(row) 

    for r in rows:
        print(r)

    return rows


def save_results_csv(csv_records, rows, results_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    results_file = os.path.join(results_dir, "results_" + timestamp + ".csv")
    summary_file = os.path.join(results_dir, "results_summary_" + timestamp + ".csv")
    if len(csv_records) > 0:
        os.makedirs(results_dir, exist_ok=True)
        with open(results_file, mode="a") as f:
            for line in csv_records:
                f.write(line+"\n")
        with open(summary_file, mode="a") as f:
            for line in rows:
                f.write(line+"\n")

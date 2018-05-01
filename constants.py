# Please specify project home dir, e.g. "/home/username/UnsupCLIR/"
HOME = ""
PROCESS_COUNT = 60

# Paths required for
PATH_EMB_BASE = HOME + "Embeddings/"
PATH_BASE_EVAL = HOME + "Data/CLEF/RelAssess/"
PATH_BASE_QUERIES = HOME + "Data/CLEF/Topics/"
PATH_BASE_DOCUMENTS = HOME + "Data/CLEF/DocumentData/"

# Paths for running ensemble experiments
RESULTS_DIR = HOME + "Results/"
RESULTS_FILE = RESULTS_DIR + "results.csv"
ENSEMBLE_RESULTS_FILE = RESULTS_DIR + "ensemble_results.csv"

YEARs = ["2001", "2002", "2003"]
METHODs = ["Conneau", "Smith", "Vulic"]  # notation in paper: ["CL-UNSUP", "CL-WT", "CD-CL"]
LANGUAGEs = [("en", "english"), ("nl", "dutch"), ("it", "italian"), ("fi", "finnish")]
LANGUAGE_PAIRS = [(LANGUAGEs[0], LANGUAGEs[1]), (LANGUAGEs[0], LANGUAGEs[2]), (LANGUAGEs[0], LANGUAGEs[3])]
LANGUAGE_PAIRS_SHORT = ["enit", "enfi", "ennl"]

# Paths for running Europarl experiments (not part of the paper)
PATH_EMB_EN = PATH_EMB_BASE + "europarl/" + "wiki.en.mapped.vec"
PATH_EMB_DE = PATH_EMB_BASE + "europarl/" + "wiki.de.mapped.vec"

PATH_MT_DE = HOME + "/Europarl/Europarl.de-en.de"
PATH_MT_EN = HOME + "/Europarl/Europarl.de-en.en"

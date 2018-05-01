# Please specify project home dir, e.g. "/home/username/UnsupCLIR/"
HOME = ""

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
METHODs = ["Conneau", "Smith", "Vulic"]  # in the paper we refer to them as ["CL-UNSUP", "CL-WT", "CD-CL"] respectively
LANGUAGEs = [("en", "english"), ("nl", "dutch"), ("it", "italian"), ("fi", "finnish")]
LANGUAGE_PAIRS = [(LANGUAGEs[0], LANGUAGEs[1]), (LANGUAGEs[0], LANGUAGEs[2]), (LANGUAGEs[0], LANGUAGEs[3])]
LANGUAGE_PAIRS_SHORT = ["enit", "enfi", "ennl"]

PROCESS_COUNT = 60

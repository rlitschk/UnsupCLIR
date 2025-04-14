PROCESS_COUNT = 20
EMBEDDING_DIM = 300

RUN_MODELS = [
  "SUM",
  "IDF-SUM",
  "LM-UNI",
  "TbT-QT",
  "MT-IR",
]

LANGUAGEs = [("en", "english"), ("de", "german"), ("it", "italian"), ("fi", "finnish"), ("ru", "russian"), ("nl", "dutch")]
short2pair = {e[0]: e for e in LANGUAGEs}

# CLEF Evaluation Campaigns
YEARS = [
  "2001",
  "2002", 
  "2003",
]

sigir18_CLWEs = [
  "Vulic", # CL-CD / BWESG
  "Smith", # CL-WT / Proc 
  "Conneau", # CL-UNSUP / Muse
]
sigir19_CLWEs = [
  "cca",
  "proc",
  "procb",
  "rcsls",
  "icp", 
  "muse",
  "vecmap"
] 
CLWEs = sigir18_CLWEs + sigir19_CLWEs

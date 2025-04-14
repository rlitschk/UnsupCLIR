EMB_DIR=Embeddings_sigir18
OUTPUT_DIR=output/sigir18-clef/

if [ -d "$EMB_DIR" ]; then
    echo "Using data in $EMB_DIR folder"
else
    echo "Embeddings directory $DIR does not exist, downloading embeddings..."
    ./download_emb_sigir18.sh
fi

python -c "import nltk;nltk.download('stopwords')"
python -c "import nltk;nltk.download('punkt')"

python experiment.py --output_dir $OUTPUT_DIR \
--dataset clef \
--clwe Vulic Smith Conneau \
--language_pairs ennl enit enfi \
--years 2001 2002 2003 \
--models LM-UNI SUM IDF-SUM TbT-QT \
--embeddings_dir $EMB_DIR \
--is_pilot 

python ensemble_clef.py --directory $OUTPUT_DIR \
--clwe Vulic Smith Conneau \
--language_pairs ennl enit enfi \
--years 2001 2002 2003 

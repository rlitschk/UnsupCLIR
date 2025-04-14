EMB_DIR=Embeddings_sigir19
OUTPUT_DIR=/output/sigir19-clef/
TRANSLATED_QUERIES_DIR=gtranslated_clef_queries/

if [ -d "$EMB_DIR" ]; then
    echo "Using data in $EMB_DIR folder"
else
    echo "Embeddings directory $DIR does not exist, downloading embeddings..."
    ./download_emb_sigir19.sh
fi

python -c "import nltk;nltk.download('stopwords')"
python -c "import nltk;nltk.download('punkt')"

python experiment.py --output_dir $OUTPUT_DIR \
--dataset clef \
--clwe cca proc procb rcsls icp muse vecmap \
--language_pairs defi deit deru ende enfi enit enru fiit firu \
--years 2003 \
--models LM-UNI MT-IR IDF-SUM TbT-QT \
--embeddings_dir $EMB_DIR \
--path_translated_queries $TRANSLATED_QUERIES_DIR

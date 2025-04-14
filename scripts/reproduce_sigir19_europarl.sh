EUROPARL_DATA_DIR=europarl
EMB_DIR=Embeddings_sigir19
OUTPUT_DIR=output/sigir19-europarl/

if [ -d "$EUROPARL_DATA_DIR" ]; then
    echo "Using data in $EUROPARL_DATA_DIR folder"
else
    echo "Directory $DIR does not exist, downloading data..."
    wget https://madata.bib.uni-mannheim.de/360/9/europarl.tar.gz
    echo "unzipping..."
    tar xzvf europarl.tar.gz
    echo "rm tar file..."
    rm europarl.tar.gz
fi

if [ -d "$EMB_DIR" ]; then
    echo "Using data in $EMB_DIR folder"
else
    echo "Embeddings directory $DIR does not exist, downloading embeddings..."
    ./download_emb_sigir19.sh
fi

python -c "import nltk;nltk.download('stopwords')"
python -c "import nltk;nltk.download('punkt')"

echo "start experiment"
python experiment.py --output_dir $OUTPUT_DIR \
--dataset europarl \
--clwe cca proc procb rcsls icp muse vecmap \
--language_pairs defi deit ende enfi enit fiit \
--models LM-UNI MT-IR IDF-SUM TbT-QT \
--embeddings_dir $EMB_DIR \
--path_translated_queries $EUROPARL_DATA_DIR

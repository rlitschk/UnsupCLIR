echo "Downloading MUSE embeddings (1/7)"
wget https://madata.bib.uni-mannheim.de/360/1/muse.tar.gz

echo "Downloading PROC embeddings (2/7)"
wget https://madata.bib.uni-mannheim.de/360/2/proc.tar.gz

echo "Downloading CCA embeddings (3/7)"
wget https://madata.bib.uni-mannheim.de/360/7/procb.tar.gz

echo "Downloading CCA embeddings (4/7)"
wget https://madata.bib.uni-mannheim.de/360/3/cca.tar.gz

echo "Downloading ICP embeddings (5/7)"
wget https://madata.bib.uni-mannheim.de/360/4/icp.tar.gz

echo "Downloading RCSLS embeddings (6/7)"
wget https://madata.bib.uni-mannheim.de/360/6/rcsls.tar.gz

echo "Downloading Vecmap embeddings (7/7)"
wget https://madata.bib.uni-mannheim.de/360/8/vecmap.tar.gz

files=("proc.tar.gz" "vecmap.tar.gz" "rcsls.tar.gz" "procb.tar.gz" "muse.tar.gz" "cca.tar.gz" "icp.tar.gz")

# unzip 
for item in "${files[@]}"
do
    # commands to be executed for each item
    echo unpacking $item
    tar -xvzf $item
    rm $item
done

mkdir Embeddings_sigir19
folders=("proc" "vecmap" "rcsls" "procb" "muse" "cca" "icp")
for item in "${folders[@]}"
do
    # commands to be executed for each item
    mv $item Embeddings_sigir19
done

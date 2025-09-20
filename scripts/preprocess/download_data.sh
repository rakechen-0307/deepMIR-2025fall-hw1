mkdir -p ./data

# Download and unzip the data
python -m src.preprocess.download_data
mv ./hw1.zip ./data/
cd ./data
unzip hw1.zip
rm hw1.zip
cd ../
#!/bin/bash
set -e  

mkdir -p dataset
cd dataset

pip install gdown librosa speechbrain jiwer
gdown 19CV4WZgYez-i2oHV2r9maJofjNqcTX4o 
gdown 1v75mLO-TVfPXe27o54JMlXD5cQ81eaVG 
gdown 1YgTF-NbHuweHWr2LahS_X9j--laGDnIK

unzip -o voices.zip

cd /
if [[ "$1" == "phoneme" ]]; then
    echo "Preprocessing for phoneme-based model"
    python workspace/conv-rnnt/utils/construct_phoneme.py
elif [[ "$1" == "char" ]]; then
    echo "Preprocessing for normal model"
    python workspace/conv-rnnt/utils/construct_char.py
else
    echo "Preprocessing for normal model"
    python workspace/conv-rnnt/utils/construct.py
mkdir workspace/conv-rnnt/saves
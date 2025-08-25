# Conv-RNNT — My Implementation

> Implementation of **ConvRNNT** (Convolutional Recurrent Neural Network Transducer) for end-to-end speech recognition, inspired by the paper:  
> **ConvRNNT: Convolutional RNN-Transducer for Efficient Speech Recognition** (2022).  
> 📄 Paper: https://arxiv.org/pdf/2209.14868

## 🔗 Repository
- GitHub: https://github.com/itsmekhoathekid/conv-rnnt

## 🚀 Quickstart

### 1) Clone & Setup
```
bash
git clone https://github.com/itsmekhoathekid/conv-rnnt
cd conv-rnnt
```

### 2) Download & Prepare Dataset
This will download the datasets configured inside the script and generate manifests/features as needed.
```
bash
bash ./prep_data.sh
```

### 3) Train
Train with a YAML/JSON config of your choice.
```
bash
python train.py --config path/to/train_config.yaml
```

### 4) Inference (example)
```
bash
python infererence.py --config path/to/train_config.yaml --epoch num_epoch
```

## 📦 Project Layout (typical)
```
conv-rnnt/
├── prep_data.sh                 # dataset download & preprocessing
├── train.py                     # training entry point
├── inference.py                     # inference script (optional)
├── configs/                     # training configs (yaml/json)
├── models/                    # model, losses, data, utils
│   ├── model.py
│   ├── encoder.py
│   ├── decoder.py
│   └── ...
├── utils/ 
└── README.md
```


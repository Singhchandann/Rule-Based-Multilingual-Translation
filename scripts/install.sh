
### **scripts/install.sh**
   Create an installation script to install all necessary dependencies:
```bash
#!/bin/bash

# Update and install necessary dependencies
sudo apt-get update
sudo apt-get install -y cmake wget unzip

# Install Python packages
pip install pip==24.0
pip install gradio python-dotenv ai4bharat-transliteration mahaNLP inference ctranslate2

# Clone necessary repositories
git clone https://github.com/AI4Bharat/IndicTrans2
cd IndicTrans2
source install.sh

# Download pre-trained models
wget https://indictrans2-public.objectstore.e2enetworks.net/it2_distilled_ckpts/indic-en.zip
mkdir indic-en
unzip indic-en.zip -d ./indic-en

wget https://indictrans2-public.objectstore.e2enetworks.net/it2_distilled_ckpts/en-indic.zip
mkdir en-indic
unzip en-indic.zip -d ./en-indic

wget https://indictrans2-public.objectstore.e2enetworks.net/it2_distilled_ckpts/indic-indic.zip
mkdir indic-indic
unzip indic-indic.zip -d ./indic-indic

# Install SentencePiece
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build && cd build
cmake .. && make -j $(nproc)
sudo make install && sudo ldconfig -v

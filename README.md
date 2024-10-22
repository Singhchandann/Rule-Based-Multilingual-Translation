# Rule-Based Multilingual Translation Using Custom Data

This project implements a **rule-based multilingual translation system** for English, Hindi, and Marathi languages, incorporating **custom shaskiya words** using custom data. The solution leverages various models from **IndicTrans2**, including named entity recognition (NER) and transliteration for person tags.

## Directory Structure
 
Rule-Based-Multilingual-Translation/  
├── inference/                    # Inference engine from IndicTrans2 (import from IndicTrans2)  
├── data/  
│   └── Final_Translation_Data.xlsx # Custom data for translations  
├── scripts/  
│   ├── install.sh                 # Installation script for dependencies  
│   └── translation.py             # Main Python script  
├── marathi_stopwords.txt          # File containing Marathi suffixes for correction  
├── indic-en/                      # Folder for Indic-EN model  
├── en-indic/                      # Folder for EN-Indic model  
├── indic-indic/                   # Folder for Indic-Indic model  
├── README.md                      # Project documentation  
├── requirements.txt               # Python dependencies  
└── LICENSE                        # License for the project  

## Folder Structure  
- `data/` — contains custom datasets (e.g., translation corrections, Marathi stopwords).  
- `inference/` — contains files cloned from [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) after installation.  
- `models/` — stores pretrained models for translation.  
- `src/` — main source files for the translation system.  
- `.env` — environment variables for your API keys or paths.  

### Key Features:  
- **Custom NER-based transliteration** for Marathi names with suffix handling.  
- **Rule-based corrections** for English-to-Marathi translations using custom datasets.  
- Supports **multiple translation directions** between English, Marathi, and Hindi.  
- **Gradio interface** for easy usage as a web-based app.  

## Demo
Check out the [Gradio Demo](#) for live multilingual translation.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Singhchandann/Rule-Based-Multilingual-Translation.git
   cd Rule-Based-Multilingual-Translation
   ```

2. Install the dependencies using the provided installation script:
   ```bash
   cd scripts
   ./install.sh
   ```

3. Download the necessary model files:

   ```bash
   # Download models for IndicTrans2
   wget https://indictrans2-public.objectstore.e2enetworks.net/it2_distilled_ckpts/indic-en.zip
   mkdir indic-en
   unzip indic-en.zip -d ./indic-en
   ```
   ```bash
   wget https://indictrans2-public.objectstore.e2enetworks.net/it2_distilled_ckpts/en-indic.zip
   mkdir en-indic
   unzip en-indic.zip -d ./en-indic
   ```
   ```bash
   wget https://indictrans2-public.objectstore.e2enetworks.net/it2_distilled_ckpts/indic-indic.zip
   mkdir indic-indic
   unzip indic-indic.zip -d ./indic-indic
   ```
   
3. Install additional dependencies:

   ```bash
   pip install -r ../requirements.txt
   ```
   
## Usage
Run the Gradio interface for multilingual translation:

   ```bash
   python scripts/translation.py
   ```

# Supported Language Pairs:
English ↔ Marathi  
English ↔ Hindi  
Hindi ↔ Marathi  

# Example
You can provide a custom translation input with the following sample structure:

   ```bash
   Input: "The government has announced a new policy"
   Source Language: English
   Target Language: Marathi
   ```

# Output:

   ```text
सरकारने एक नवीन धोरण जाहीर केले आहे
   ```

## How It Works  

1. Named Entity Recognition (NER): The system identifies person names in Marathi and handles them separately during translation, ensuring correct transliteration and context preservation.
2. Custom Data: The system uses custom data from Final_Translation_Data.xlsx to correct translation errors and ensure accurate output.
3. Transliteration: The system transliterates names and person tokens for better contextual accuracy.
4. Rule-Based Correction: Translations are further corrected using a set of predefined rules, ensuring improved accuracy.

## File Descriptions
* scripts/translation.py: Main script for handling translation, NER, transliteration, and corrections.
* data/Final_Translation_Data.xlsx: Custom dataset used for fine-tuning translations with common shaskiya words.

## Dependencies
* The project requires the following Python packages:  

gradio  
huggingface_hub  
ai4bharat-transliteration  
indicnlp  
mahaNLP  
ctranslate2  
nltk  

## License  

This project is licensed under the MIT License - see the LICENSE file for details.  

## Acknowledgments  
IndicTrans2 for providing pre-trained models for multilingual translation.

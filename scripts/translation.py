#replace installed inference folder with inference folder of IndicTrans2
import shutil
import os

# Source and destination paths
source_folder = "/content/IndicTrans2/inference"
destination_folder = "/usr/local/lib/python3.10/dist-packages"

# Get the folder name from the source path
folder_name = os.path.basename(source_folder)

# Create the new destination path with the folder name
new_destination_path = os.path.join(destination_folder, folder_name)

# Remove the destination folder if it exists
if os.path.exists(new_destination_path):
    shutil.rmtree(new_destination_path)

# Move the folder
shutil.copytree(source_folder, new_destination_path)
# Import necessary libraries
import requests
from dotenv import load_dotenv
import os
import gradio as gr
import pandas as pd
from mahaNLP.tagger import EntityRecognizer
from inference.engine import Model
from ai4bharat.transliteration import XlitEngine
import nltk
nltk.download('punkt')

# Initialize models
model = Model(r"/content/indic-en/ct2_fp16_model", model_type="ctranslate2")
model2 = EntityRecognizer()
model4 = Model(r"/content/en-indic/ct2_fp16_model", model_type="ctranslate2")
model5 = Model(r"/content/indic-indic/ct2_fp16_model", model_type="ctranslate2")
e = XlitEngine(beam_width=10, src_script_type="indic")

# Function to load Marathi suffixes from file
def load_marathi_suffixes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        suffixes = [line.strip() for line in file]
    return suffixes

marathi_suffixes = load_marathi_suffixes(r"/content/drive/MyDrive/marathi_stopwords.txt")

def get_suffix(word, suffixes):
    for suffix in suffixes:
        if word.endswith(suffix):
            main_word = word[:-len(suffix)].strip()
            return main_word, suffix
    return word, ''

# Function to perform Named Entity Recognition (NER) and handle suffixes separately
def ner_tagger(text, suffixes):
    tag = model2.get_token_labels(text)
    tokens = [(row.word, row.entity_group) for row in tag.itertuples(index=False)]
    combined_tokens = []
    for word, entity in tokens:
        if entity == "Person":
            main_word, suffix = get_suffix(word, suffixes)
            combined_tokens.append((main_word, "Person"))
            if suffix:
                combined_tokens.append((suffix, "Other"))
        else:
            combined_tokens.append((word, entity))
    return combined_tokens

# Function to transliterate person tokens
def transliterate_person_tokens(tokens):
    transliterated_tokens = []
    for token, label in tokens:
        if label == 'Person':
            split_token = token.rsplit(' ', 1)
            if len(split_token) > 1:
                main_name, suffix = split_token
            else:
                main_name = split_token[0]
                suffix = ''
            transliterated_main_name = e.translit_sentence(main_name, 'mr')
            transliterated_token = transliterated_main_name + (' ' + suffix if suffix else '')
            transliterated_tokens.append((transliterated_token, label))
        else:
            transliterated_tokens.append((token, label))
    return transliterated_tokens

# Function to transliterate only person tags and maintain their positions
def transliterate_person_tags_only(text, suffixes):
    # Perform Named Entity Recognition (NER)
    tokens = ner_tagger(text, suffixes)

    # Transliterate person tags only
    transliterated_text = []
    original_person_tokens = {}  # To store the transliterated person tokens and their original positions
    index_offset = 0  # Offset for adjusting index when inserting placeholders
    for index, (token, label) in enumerate(tokens):
        if label == 'Person':
            # Transliterate the token
            transliterated_token = transliterate_person_tokens([(token, label)])
            original_person_tokens[index] = transliterated_token[0][0]  # Store transliterated token and original position
            transliterated_text.append(f"[PERSON{index}]")  # Add a placeholder for the transliterated person token
            index_offset += 1  # Increase offset after inserting a placeholder
        else:
            transliterated_text.append(token)

    return transliterated_text, original_person_tokens


def count_person_tags(text, suffixes):
    # Perform Named Entity Recognition (NER)
    tokens = ner_tagger(text, suffixes)

    # Count the number of person tags
    person_count = sum(1 for token, label in tokens if label == 'Person')

    return person_count


def process_text(text, src_lang, tgt_lang, suffixes):
    # Count the number of person tags
    num_person_tags = count_person_tags(text, suffixes)

    if num_person_tags > 6:
        # Translate the text directly
        translated_text = model.batch_translate([text], src_lang, tgt_lang)[0]
    else:
        # Transliterate person tags only
        transliterated_text, original_person_tokens = transliterate_person_tags_only(text, suffixes)

        # Translate the transliterated text
        translated_text = model.batch_translate([' '.join(transliterated_text)], src_lang, tgt_lang)[0]

        # Replace the placeholders with original person tokens in their original positions
        for index, transliterated_token in original_person_tokens.items():
            translated_text = translated_text.replace(f"[PERSON{index}]", transliterated_token, 1)

    return translated_text


def translate_sentence_with_replacements(model, df, sentence):
    # Translate the original sentence
    translated_sentence = model4.batch_translate([sentence], "eng_Latn", "mar_Deva")[0]

    # Tokenize the original sentence
    sentence_tokens = sentence.lower().split()

    # Find all rows where eng_Latn phrases match as whole phrases in the original sentence
    mask = df['eng_Latn'].apply(lambda x: all(word in sentence_tokens for word in x.lower().split()))
    filtered_df = df[mask]

    # Sort filtered DataFrame by length of mar_Deva_wrong in descending order
    filtered_df = filtered_df.sort_values(by='mar_Deva_wrong', key=lambda x: x.str.len(), ascending=False)

    # Store replacements
    replacements = {}
    for index, row in filtered_df.iterrows():
        mar_wrong_word = row['mar_Deva_wrong']
        mar_correct_word = row['mar_Deva']
        if isinstance(mar_wrong_word, str) and isinstance(mar_correct_word, str):
            if mar_wrong_word in translated_sentence and mar_wrong_word not in replacements:
                translated_sentence = translated_sentence.replace(mar_wrong_word, mar_correct_word)
                replacements[mar_wrong_word] = mar_correct_word

    return translated_sentence

# Read the DataFrame
df1 = pd.read_excel(r"/Final_Translation_Data.xlsx")


# Function to translate Marathi to English
def translate_marathi_to_english(input_text):
    translated_text_en = process_text(input_text, "mar_Deva", "eng_Latn", marathi_suffixes)
    return translated_text_en

# Define the translation function for English to Marathi
def translate_english_to_marathi(input_text):
    translated_text_mr = translate_sentence_with_replacements(model4, df1, input_text)
    return translated_text_mr

# Define the translation function for English to Hindi
def translate_english_to_hindi(input_text):
    translated_text_hi = model4.translate_paragraph(input_text, "eng_Latn", "hin_Deva")
    return translated_text_hi

# Define the translation function for Hindi to English
def translate_hindi_to_english(input_text):
    translated_text_en = model.translate_paragraph(input_text, "hin_Deva", "eng_Latn")
    return translated_text_en

# Define the translation function for Hindi to Marathi
def translate_hindi_to_marathi(input_text):
    translated_text_mr = model5.translate_paragraph(input_text, "hin_Deva", "mar_Deva")
    return translated_text_mr

# Define the translation function for Hindi to Marathi
def translate_marathi_to_hindi(input_text):
    translated_text_hi = model5.translate_paragraph(input_text, "mar_Deva", "hin_Deva")
    return translated_text_hi

# Define the translation function for Gradio
def translate_with_gradio(input_text, src_lang, tgt_lang):
    if src_lang == "Marathi" and tgt_lang == "English":
        return translate_marathi_to_english(input_text)
    elif src_lang == "English" and tgt_lang == "Marathi":
        return translate_english_to_marathi(input_text)
    elif src_lang == "English" and tgt_lang == "Hindi":
        return translate_english_to_hindi(input_text)
    elif src_lang == "Hindi" and tgt_lang == "English":
        return translate_hindi_to_english(input_text)
    elif src_lang == "Hindi" and tgt_lang == "Marathi":
        return translate_hindi_to_marathi(input_text)
    elif src_lang == "Marathi" and tgt_lang == "Hindi":
        return translate_marathi_to_hindi(input_text)
    else:
        return "ERROR"

languages = ['English', 'Marathi', 'Hindi']
# Create the Gradio interface
demo = gr.Interface(
    fn=translate_with_gradio,
    inputs=[
        gr.Text(label="Enter text"),
        gr.Dropdown(label="From",choices=languages,value="English",),
        gr.Dropdown(label="To",choices=languages,value="Marathi")
    ],
    outputs=gr.Textbox(label="Translation"),
    title="Multilingual Translation with shaskiya words",
    description="Translation With shaskiya Words In  Marathi",
)

# Launch the interface
demo.launch(share=True)

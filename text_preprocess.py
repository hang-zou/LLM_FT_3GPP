import os, shutil
import re
from bs4 import BeautifulSoup
import nltk.tokenize

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

def remove_urls(text):
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

#def remove_special_chars(text):
    #return re.sub(r'[^\w\s]', ' ', text)

def preprocess_text(text):
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = re.sub(r'[\n, \t]', ' ', text)
    #text = remove_special_chars(text)
    return text

def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as html_file:
        html_content = html_file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    for s in soup('table'):
        s.extract()

    paragraphs = soup.find_all(['h1', 'p'])
    new_paragraphs = []
    current_paragraph = ""
    found_introduction = False
    stop_processing = False
    attr_ignore = ['header', 'footer', 'tF', 'pL'] # header, footer, figure caption, pseudo code
    text_ignore = ['///', 'Page', 'C2']

    for paragraph in paragraphs:

        ignore_paragraph = False
        paragraph_len = 50

        if paragraph.name == 'h1':
            if 'Introduction' in paragraph.get_text():
                found_introduction = True
            elif 'References' in paragraph.get_text() or 'references' in paragraph.get_text():
                stop_processing = True

        elif found_introduction and paragraph.name == 'p' and not stop_processing:

            paragraph_text = paragraph.get_text()

            if paragraph.has_attr('class'):
                if paragraph['class'][0] in attr_ignore:
                    ignore_paragraph = True

            for text in text_ignore:
                if text in paragraph_text:
                    ignore_paragraph = True

            if ignore_paragraph:
                continue
                
            sentences = nltk.tokenize.sent_tokenize(paragraph_text)
            tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            for s in sentences:
                current_paragraph_t = tokenizer.tokenize(current_paragraph)
                s_t = tokenizer.tokenize(s)
                if s.isupper():
                    continue
                if len(current_paragraph_t) + len(s_t) < paragraph_len:
                    current_paragraph += s
                else:
                    new_paragraphs.append(current_paragraph.strip())
                    current_paragraph = s

    if current_paragraph:
        new_paragraphs.append(current_paragraph.strip())

    return new_paragraphs


def should_ignore(text):
    ignore_phrases = [ 'CHANGE REQUEST', 'change request']
    return any(phrase in text for phrase in ignore_phrases)

def split_paragraphs(paragraphs):
    new_paragraphs = []
    for text in paragraphs:
        paragraphs = re.split(r'\n+', text)
        new_paragraphs.extend(paragraphs)
    return new_paragraphs

#input_folder = '/efs/hang/telecombrain/globecom23/3GPP_NEW_TINY/RAN3'
#output_folder = '/efs/hang/telecombrain/globecom23/Paragraphs/NEW/RAN3'
input_folder = '/efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/NEW'
output_base_folder = '/efs/hang/telecombrain/globecom23/Paragraphs/4G_050T/NEW'

working_groups = ["CT1", "CT3", "CT4", "CT6", 
                  "RAN1", "RAN2", "RAN3", "RAN4", "RAN5", 
                  "SA1", "SA2", "SA3", "SA4", "SA5", "SA6"]
#working_groups = ["RAN4", "RAN5", 
                  #"SA1", "SA2", "SA3", "SA4", "SA5", "SA6"]
working_groups_limited = ["RAN1", "RAN2", "RAN4"]
#working_groups = [ "RAN2"]



max_paragraph_num = 4  # Set the desired maximum number of paragraphs per file

for working_group in working_groups:
    input_group_folder = os.path.join(input_folder, working_group)
    output_group_folder = os.path.join(output_base_folder, working_group)

    if os.path.exists(output_group_folder):
        shutil.rmtree(output_group_folder)

    os.makedirs(output_group_folder)

    for file_name in os.listdir(input_group_folder):

        if file_name.endswith('.html'):
            input_file_path = os.path.join(input_group_folder, file_name)

            with open(input_file_path, 'r', encoding='utf-8') as input_file:
                raw_text = input_file.read()

            if should_ignore(raw_text):
                continue

            paragraphs = extract_text_from_html(input_file_path)
            paragraphs = [preprocess_text(paragraph) for paragraph in paragraphs]

            paragraph_counter = 0
            for i, paragraph in enumerate(paragraphs):
                if paragraph_counter >= max_paragraph_num and working_group in working_groups_limited:
                    break

                output_file_name = f'{file_name[:-4]}_{i}.txt'
                with open(os.path.join(output_group_folder, output_file_name), 'w', encoding='utf-8') as output_file:
                    output_file.write(paragraph)
                paragraph_counter += 1

    print(f"All files of working group {working_group} are split into paragraphs!\n\n")




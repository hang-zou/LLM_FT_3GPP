import os, shutil
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import nltk.tokenize
from sklearn.model_selection import train_test_split
import json
import random

# cd /efs/hang/telecombrain/globecom23/BERT_FT_3GPP

def build_dataset(input_folder, portion=1.0):
    data = pd.DataFrame(columns=['text', 'class', 'length'])
    selected_classes = {'CT1', 'CT3', 'CT4', 'CT6',
                        'RAN1','RAN2','RAN3','RAN4','RAN5',
                        'SA1','SA2','SA3','SA4','SA5','SA6'}
    #selected_classes = {'CT1', 
                        #'RAN1','RAN2',
                        #'SA1'}
    for class_name in os.listdir(input_folder):
        if class_name in selected_classes:
            class_path = os.path.join(input_folder, class_name)

            if os.path.isdir(class_path):
                all_files = os.listdir(class_path)
                num_files = len(all_files)
                portion_files = int(num_files * portion)

                # Shuffle the list of files to ensure randomness
                random.shuffle(all_files)

                for file_name in all_files[:portion_files]:
                    file_path = os.path.join(class_path, file_name)

                    if os.path.isfile(file_path):
                        with open(file_path, 'r', encoding='utf-8') as file:
                            file_content = file.read()
                            tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
                            file_content_t = tokenizer.tokenize(file_content)
                            data = data.append({'text': file_content, 'class': class_name, 'length': len(file_content_t)}, ignore_index=True)
                print(f"\n\n\n{class_name} is OK!\n\n")

    return data



def one_hot_encode_labels(data):
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_labels = one_hot_encoder.fit_transform(data['class'].values.reshape(-1, 1))
    one_hot_labels_df = pd.DataFrame(one_hot_labels, columns=one_hot_encoder.get_feature_names_out(['class']))

    return pd.concat([data['text'], one_hot_labels_df, data['length']], axis=1)

def split_dataset(data, train_size):
    train, val = train_test_split(data, train_size=train_size, random_state=42)
    return train, val



def main():
    input_folder_train_val = '/efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD'
    input_folder_test = '/efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE'
    output_folder = '/efs/hang/telecombrain/globecom23/Dataset/5G_2020_200FT_200TEST_100PER'
    output_json = '3GPP_{}.json'

    # Set the portion of the train_val_data to select
    train_val_portion = 100/100 
    # Set the portion of the training part of the dataset
    train_size = 0.8
    # generate training and validation part?
    skip_train_val = True
    # generate test part?
    skip_test = False

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    # Only build the training and validation dataset if the flag is False
    if not skip_train_val:
        # Build the training and validation dataset
        if train_val_portion == 1:
            train_val_data = build_dataset(input_folder_train_val)
        else:
            train_val_data = build_dataset(input_folder_train_val,train_val_portion)
        one_hot_train_val_data = one_hot_encode_labels(train_val_data)
        train_data, val_data = split_dataset(one_hot_train_val_data, train_size)
    else:
        train_data, val_data = None, None

    # Only build the test dataset if the flag is False
    if not skip_test:
        test_data = build_dataset(input_folder_test)
        one_hot_test_data = one_hot_encode_labels(test_data)
    else:
        one_hot_test_data = None

    for split_name, split_data in {'train': train_data, 'validation': val_data, 'test': one_hot_test_data}.items():
        if split_data is not None:
            output_path = os.path.join(output_folder, output_json.format(split_name))
            with open(output_path, 'w') as outfile:
                json.dump(split_data.to_dict(orient='records'), outfile)

    print("Annotation and splitting for 3GPP files are finished!")

if __name__ == '__main__':
    main()


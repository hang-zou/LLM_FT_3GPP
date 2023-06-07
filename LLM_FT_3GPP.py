
# # Fine-tuning BERT (and friends) for 3GPP file classification
# In this notebook, we are going to fine-tune BERT to predict one or more labels 
# of 3GPP files for a given piece of text. 
# Note that this notebook illustrates how to fine-tune a bert-base-uncased model, 
# but you can also fine-tune a RoBERTa, DeBERTa, DistilBERT, CANINE, ... checkpoint in the same way. 

# All of those work in the same way: they add a linear layer on top of the base model, 
# which is used to produce a tensor of shape (batch_size, num_labels), 
#indicating the unnormalized scores for a number of labels for every example in the batch.
##--------------------------------------------------
# ## Set-up environment
# First, we install the libraries which we'll use: HuggingFace Transformers and Datasets.
#!pip install -q transformers datasets
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction
from datasets import load_dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import wandb



##--------------------------------------------------
# ## Load dataset
wandb.login(key="089114d20a5448f90191fd2db8212aecafdea332")

data_files = {
    'train': '/efs/hang/telecombrain/globecom23/Dataset/5G_200FT_200TEST_80PER/3GPP_train.json',
    'validation': '/efs/hang/telecombrain/globecom23/Dataset/5G_200FT_200TEST_80PER/3GPP_validation.json',
    'test': '/efs/hang/telecombrain/globecom23/Dataset/5G_200FT_200TEST_80PER/3GPP_test.json'
}
dataset = load_dataset("json", data_files=data_files)


# As we can see, the dataset contains 3 splits: one for training, one for validation and one for testing.
dataset
# The dataset consists of parapraphs extracted from 3GPP files, 
# labeled with unique working group, together with its length.
# Let's create a list that contains the labels, as well as 2 dictionaries that map labels to integers and back.
labels = [label for label in dataset['train'].features.keys() if label not in ['text', 'length']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
labels

##--------------------------------------------------
# ## Preprocess data
# 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
#tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["text"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

##--------------------------------------------------
# ## Define model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)

##--------------------------------------------------
# ## Train the model!
# set up batch size
batch_size = 32
# set up performance metric
metric_name = "f1"

args = TrainingArguments(
    f"bert-finetuned-3gpp/5G_200FT_200TEST_80PER_r4",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3, # if portion is greater than 50%, set to 3
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)

# We are also going to compute metrics while training. 
# For this, we need to define a `compute_metrics` function, 
# that returns a dictionary with the desired metric values.
def single_label_metrics(predictions, labels):
    # Apply softmax on predictions which are of shape (batch_size, num_labels)
    softmax = torch.nn.Softmax()
    probs = softmax(torch.Tensor(predictions))
    one_hot_encoder = OneHotEncoder(sparse=False)
    max_index = np.argmax(probs.numpy(),axis=1)
    #y_pred = one_hot_encoder.fit_transform(max_index.reshape(-1, 1))
    y_pred = one_hot_encoder.fit(np.arange(labels.shape[1]).reshape(-1, 1)).transform(max_index.reshape(-1, 1))

    # Compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    
    # Return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = single_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result



# Let's start training!
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#test_len = len(encoded_dataset["test"])

# BERT without fine-tuning
initial_results = trainer.evaluate(encoded_dataset["test"])

# training the model!
trainer.train()

##--------------------------------------------------
# ## Evaluate
# After training, we evaluate our model on the test set.
final_results = trainer.evaluate(encoded_dataset["test"])

# Compare the results
print("\n\nPerformance of BERT before fine-tuning on 3GPP files:")
print(initial_results)

print("\nPerformance of BERT after fine-tuning on 3GPP files:")
print(final_results)




from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn import feature_selection
import argparse
from sklearn import feature_selection

LR = 2e-5
EPOCHS = 20
BATCH_SIZE = 32
MODEL = "cardiffnlp/twitter-xlm-roberta-base" # use this to finetune the language model
MAX_TRAINING_EXAMPLES = -1 # set this to -1 if you want to use the whole training set

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
        
def args():
    parser = argparse.ArgumentParser(description='Tweet Intimacy')
    parser.add_argument('--dataset', type=str, help='path to dataset directory')
    return parser.parse_args()

def compute_metrics(eval_preds):
    predictions, label_ids = eval_preds
    x = feature_selection.r_regression(predictions, label_ids)
    return {'accuracy': float(x[0])}

def main():

    path = args().dataset.strip('/')

    dataset_dict = {}
    for i in ['train','val','test']:
        dataset_dict[i] = {}
    for j in ['text','labels']:
        dataset_dict[i][j] = open(f"{path}/{i}_{j}.txt").read().split('\n')
        if j == 'labels':
            dataset_dict[i][j] = [float(x) for x in dataset_dict[i][j]]


    if MAX_TRAINING_EXAMPLES > 0:
        dataset_dict['train']['text']=dataset_dict['train']['text'][:MAX_TRAINING_EXAMPLES]
        dataset_dict['train']['labels']=dataset_dict['train']['labels'][:MAX_TRAINING_EXAMPLES]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    train_encodings = tokenizer(dataset_dict['train']['text'], truncation=True, padding=True)
    val_encodings = tokenizer(dataset_dict['val']['text'], truncation=True, padding=True)
    test_encodings = tokenizer(dataset_dict['test']['text'], truncation=True, padding=True)

    train_dataset = MyDataset(train_encodings, dataset_dict['train']['labels'])
    val_dataset = MyDataset(val_encodings, dataset_dict['val']['labels'])
    test_dataset = MyDataset(test_encodings, dataset_dict['test']['labels'])

    training_args = TrainingArguments(
        output_dir='./results',                   # output directory
        num_train_epochs=EPOCHS,                  # total number of training epochs
        per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
        per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
        warmup_steps=100,                         # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                        # strength of weight decay
        logging_dir='./logs',                     # directory for storing logs
        logging_steps=10,                         # when to print log
        load_best_model_at_end=True,              # load or not best model at the end
        save_strategy='epoch',
        evaluation_strategy="epoch",
        metric_for_best_model='accuracy',
    )

    num_labels = len(set(dataset_dict["train"]["labels"]))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, problem_type="regression", num_labels = 1)

    trainer = Trainer(
        model=model,                              # the instantiated 🤗 Transformers model to be trained
        args=training_args,                       # training arguments, defined above
        train_dataset=train_dataset,              # training dataset
        eval_dataset=val_dataset,                 # evaluation dataset
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model("./results/best_model") # save best model

    test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
    x = feature_selection.r_regression(test_preds_raw, test_labels)

    print("Pearson's R: ", x[0])

if __name__ == '__main__':
    main()

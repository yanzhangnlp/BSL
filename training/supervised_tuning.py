from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Training parameters
model_name = 'bert-base-nli-mean-tokens'
train_batch_size = 64
num_epochs = 1
max_seq_length = 64
moving_average_decay = 0.9999
nli_dataset_path = 'data/AllNLI'

# Save path to store our model
model_save_path = 'output/BSL_tuning-{}-{}-{}'.format(model_name, train_batch_size, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'data/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

nli_dataset_path = 'data/AllNLI.tsv.gz'
if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

model = SentenceTransformer(model_name)

# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read AllNLI train dataset")

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = []
count = 0
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'train':
            # label_id = label2int[row['label']]
            if row['label'] == "entailment":
                count += 1
                train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']]))



# Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        if row['split'] == 'dev':
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        elif row['split'] == 'test':
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.BYOLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), moving_average_decay=moving_average_decay)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
evaluation_steps = int(len(train_dataloader) * 0.1) #Evaluate every 10% of the data
logging.info("Training sentences: {}".format(len(train_samples)))
logging.info("Warmup-steps: {}".format(warmup_steps))
logging.info("Performance before training")
dev_evaluator(model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          optimizer_params={'lr': 1e-4},
          use_amp=True        #Set to True, if your GPU supports FP16 cores
          )

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################


model = SentenceTransformer(model_save_path)
test_evaluator(model, output_path=model_save_path)

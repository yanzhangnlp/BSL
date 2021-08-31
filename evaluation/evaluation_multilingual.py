from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader
import logging
import sys
import os
import torch
import numpy as np
import zipfile
import io
script_folder_path = os.path.dirname(os.path.realpath(__file__))


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name = 'stsb-xlm-r-multilingual'
# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
model = SentenceTransformer(model_name)


# Read the dataset
source_languages = ['en']                     
target_languages = ['en', 'de', 'es', 'fr', 'ar', 'tr']    
sts_corpus = "../training/data/STS2017-extended.zip" 

logging.info("Read STS test dataset")
##### Read cross-lingual Semantic Textual Similarity (STS) data ####
all_languages = list(set(list(source_languages)+list(target_languages)))
sts_data = {}
evaluators = [] 
#Open the ZIP File of STS2017-extended.zip and check for which language combinations we have STS data
with zipfile.ZipFile(sts_corpus) as zip:
        filelist = zip.namelist()
        for i in range(len(all_languages)):
                for j in range(i, len(all_languages)):
                        lang1 = all_languages[i]
                        lang2 = all_languages[j]
                        filepath = 'STS2017-extended/STS.{}-{}.txt'.format(lang1, lang2)
                        if filepath not in filelist:
                                lang1, lang2 = lang2, lang1
                                filepath = 'STS2017-extended/STS.{}-{}.txt'.format(lang1, lang2)

                        if filepath in filelist:
                                filename = os.path.basename(filepath)
                                sts_data[filename] = {'sentences1': [], 'sentences2': [], 'scores': []}

                                fIn = zip.open(filepath)
                                for line in io.TextIOWrapper(fIn, 'utf8'):
                                        sent1, sent2, score = line.strip().split("\t")
                                        score = float(score)
                                        sts_data[filename]['sentences1'].append(sent1)
                                        sts_data[filename]['sentences2'].append(sent2)
                                        sts_data[filename]['scores'].append(score)

# model = SentenceTransformer(model_save_path)
for filename, data in sts_data.items():
        test_evaluator = EmbeddingSimilarityEvaluator(data['sentences1'], data['sentences2'], data['scores'], batch_size=16, name=filename, show_progress_bar=False)
        test_evaluator(model)




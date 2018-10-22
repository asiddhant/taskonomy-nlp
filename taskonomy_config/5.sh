#!/usr/bin/env bash 
source activate allennlp
NER_TRAIN_DATA_PATH=datasets/conll2003_ner/en/train/train.txt NER_TEST_A_PATH=datasets/conll2003_ner/en/val/valid.txt  NER_TEST_B_PATH=datasets/conll2003_ner/en/test/test.txt  allennlp train -s saved_models/5 taskonomy_config/5.jsonnet


source activate allennlp
NER_TRAIN_DATA_PATH=datasets/conll-formatted-ontonotes-5.0/data/train/ NER_TEST_A_PATH=datasets/conll-formatted-ontonotes-5.0/data/development/   allennlp train -s saved_models/ner_on_wb_1k taskonomy_config/ner_on_wb_1k.jsonnet

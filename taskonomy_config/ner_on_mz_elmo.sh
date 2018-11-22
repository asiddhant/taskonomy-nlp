
NER_TRAIN_DATA_PATH=datasets/conll-formatted-ontonotes-5.0/data/train NER_TEST_A_PATH=datasets/conll-formatted-ontonotes-5.0/data/development/   allennlp train -s saved_models/ner_on_mz_elmo taskonomy_config/ner_on_mz_elmo.jsonnet


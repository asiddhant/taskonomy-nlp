
source activate allennlp
NER_TRAIN_DATA_PATH=datasets/cross-domain-subsets-ner/wb_5k.pkl NER_TEST_A_PATH=datasets/conll-formatted-ontonotes-5.0/data/development/   allennlp train -s saved_models/ner_on_wb_5k_elmo_ours taskonomy_config/ner_on_wb_5k_elmo_ours.jsonnet

source activate allennlp
SRL_TRAIN_DATA_PATH=datasets/cross-domain-subsets/wb_1k.pkl SRL_VAL_DATA_PATH=datasets/conll-formatted-ontonotes-5.0/data/development/ allennlp fine-tune -s saved_models/wb_5k_glove_50 -c taskonomy_config/wb_5k_glove.jsonnet -m saved_models/wb_5k_glove

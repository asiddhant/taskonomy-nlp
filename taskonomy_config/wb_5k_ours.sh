source activate allennlp
SRL_TRAIN_DATA_PATH=datasets/cross-domain-subsets/wb_5k.pkl SRL_VAL_DATA_PATH=datasets/conll-formatted-ontonotes-5.0/data/development/ allennlp train -s saved_models/wb_5k_ours taskonomy_config/wb_5k_ours.jsonnet

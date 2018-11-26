source activate allennlp
SRL_TRAIN_DATA_PATH=datasets/cross-task-subsets/srl_onto_5k.pkl SRL_VAL_DATA_PATH=datasets/conll-formatted-ontonotes-5.0/data/development/ allennlp train -s saved_models/srl_5k_glove_wt
taskonomy_config/srl_5k_glove_wt.jsonnet
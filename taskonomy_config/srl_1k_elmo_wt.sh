source activate allennlp
SRL_TRAIN_DATA_PATH=datasets/cross-task-subsets/srl_onto_1k.pkl SRL_VAL_DATA_PATH=datasets/conll-formatted-ontonotes-5.0/data/development/ allennlp train -s saved_models/srl_1k_elmo_wt taskonomy_config/srl_1k_elmo_wt.jsonnet

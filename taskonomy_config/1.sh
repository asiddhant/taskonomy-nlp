source activate allennlp
SRL_TRAIN_DATA_PATH=datasets/conll-formatted-ontonotes-5.0/data/train/ SRL_VAL_DATA_PATH=datasets/conll-formatted-ontonotes-5.0/data/development/ allennlp train -s saved_models/1 taskonomy_config/1.jsonnet

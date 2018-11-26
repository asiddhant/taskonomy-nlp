{
    "dataset_reader": {
        "type": "ontonotes_ner_pkl"
    },
    "iterator": {
        "type": "basic",
        "batch_size": 16
    },
    "validation_iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "tokens",
                "num_tokens"
            ]
        ],
        "batch_size": 128
    },
    "model": {
        "type": "crf_tagger",
        "calculate_span_f1": true,
        "constrain_crf_decoding": true,
        "dropout": 0.5,
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.5,
            "hidden_size": 200,
            "input_size": 100 + 128 + 200,
            "num_layers": 1
        },
        "include_start_end_transitions": false,
        "label_encoding": "BIOUL",
        "text_field_embedder": {
            "type": "weighted_average",
            "output_dim": 200,
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 100,
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                    "trainable": true
                },
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": 16
                    },
                    "encoder": {
                        "type": "cnn",
                        "conv_layer_activation": "relu",
                        "embedding_dim": 16,
                        "ngram_filter_sizes": [
                            3
                        ],
                        "num_filters": 128
                    }
                }
            }
        }
    },
    "train_data_path": std.extVar('NER_TRAIN_DATA_PATH'),
    "validation_data_path": std.extVar('NER_TEST_A_DATA_PATH'),,
    "trainer": {
        "cuda_device": 0,
        "grad_norm": 5,
        "num_epochs": 75,
        "num_serialized_models_to_keep": 3,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "patience": 25,
        "validation_metric": "+f1-measure-overall"
    },
    "validation_dataset_reader": {
        "type": "ontonotes_ner",
        "coding_scheme": "BIOUL",
        "domain_identifier": "wb",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            },
            "token_characters": {
                "type": "characters"
            },
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    }
}
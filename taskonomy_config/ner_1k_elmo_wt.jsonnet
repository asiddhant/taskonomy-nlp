{
    "dataset_reader": {
        "type": "ontonotes_ner_pkl"
    },
    "iterator": {
        "type": "basic",
        "batch_size": 8
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
            "input_size": 1024 + 128 + 200,
            "num_layers": 1
        },
        "include_start_end_transitions": false,
        "label_encoding": "BIOUL",
        "text_field_embedder": {
            "type": "weighted_average",
            "output_dim": 200,
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "do_layer_norm": false,
                    "dropout": 0.1,
                    "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
                },
                "dependency_embedder": {
                    "type": "dependency_embedder",
                    "serialization_dir":"pretrained/dependency-parser/",
                    "cuda_device":0
                },
                "constituency_embedder": {
                    "type": "constituency_embedder",
                    "serialization_dir":"pretrained/constituency-parser/",
                    "cuda_device":0
                },
                "srl_embedder": {
                    "type": "srl_embedder",
                    "serialization_dir":"pretrained/srl/",
                    "cuda_device":0
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
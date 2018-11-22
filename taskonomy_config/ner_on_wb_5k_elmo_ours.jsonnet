{
  "dataset_reader":{
    "type": "ontonotes_ner_pkl"
  },
  "validation_dataset_reader": {
    "type": "ontonotes_ner",
    "domain_identifier" : "wb",
    "coding_scheme": "BIOUL",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters"
      },
	"elmo": {
                "type": "elmo_characters"
            }
    }
  },
  "train_data_path": std.extVar("NER_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("NER_TEST_A_PATH"),
  "model": {
    "type": "crf_tagger",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
            "type": "weighted_average",
            "output_dim": 200,
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                    "do_layer_norm": false,
                    "dropout": 0.1
                },
                "ner_embedder_nw": {
                    "type": "ner_embedder",
                    "serialization_dir":"pretrained/ner_on_nw_elmo",
                    "cuda_device":0
                },
                "ner_embedder_bc": {
                    "type": "ner_embedder",
                    "serialization_dir":"pretrained/ner_on_bc_elmo",
                    "cuda_device":0
                },
                "ner_embedder_bn": {
                    "type": "ner_embedder",
                    "serialization_dir":"pretrained/ner_on_bn_elmo",
                    "cuda_device":0
                },
                "ner_embedder_mz": {
                    "type": "ner_embedder",
                    "serialization_dir":"pretrained/ner_on_mz_elmo",
                    "cuda_device":0
                },
                "ner_embedder_tc": {
                    "type": "ner_embedder",
                    "serialization_dir":"pretrained/ner_on_tc_elmo",
                    "cuda_device":0
                },
                "token_characters": {
                        "type": "character_encoding",
                        "embedding": {
                            "embedding_dim": 16
                        },
                        "encoder": {
                            "type": "cnn",
                            "embedding_dim": 16,
                            "num_filters": 128,
                            "ngram_filter_sizes": [3],
                            "conv_layer_activation": "relu"
                        }
                 }
            },
        },
    "encoder": {
        "type": "lstm",
        "input_size": 200 + 1024 + 128,
        "hidden_size": 200,
        "num_layers": 1,
        "dropout": 0.5,
        "bidirectional": true
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 16
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  }
}

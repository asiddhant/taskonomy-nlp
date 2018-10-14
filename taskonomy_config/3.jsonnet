// Our method for SRL on Ontonotes bn dataset.
{
    "dataset_reader": {
        "type": "srl",
        "domain_identifier":"bn"},
    
    "train_data_path": std.extVar('SRL_TRAIN_DATA_PATH'),
    "validation_data_path": std.extVar('SRL_VAL_DATA_PATH'),
    
    "model": {
        "type": "srl",
        "text_field_embedder": {
            "type": "weighted_average",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 100,
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                    "trainable": true
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
                "ner_embedder": {
                    "type": "ner_embedder",
                    "serialization_dir":"pretrained/ner-tagger/",
                    "cuda_device":0
                }
            },
            "output_dim": 512
        },
        "initializer": [
            [
                "tag_projection_layer.*weight",
                {
                    "type": "orthogonal"
                }
            ]
        ],
        // NOTE: This configuration is correct, but slow.
        // If you are interested in training the SRL model
        // from scratch, you should use the 'alternating_lstm_cuda'
        // encoder instead.
        "encoder": {
            "type": "alternating_lstm",
            "input_size": 712,
            "hidden_size": 300,
            "num_layers": 8,
            "recurrent_dropout_probability": 0.1,
            "use_highway": true,
            "use_input_projection_bias": false
        },
        "binary_feature_dim": 100,
        "regularizer": [
            [
                ".*scalar_parameters.*",
                {
                    "type": "l2",
                    "alpha": 0.001
                }
            ]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "tokens",
                "num_tokens"
            ]
        ],
        "batch_size": 32
    },
    "trainer": {
        "num_epochs": 25,
        "grad_clipping": 1.0,
        "patience": 200,
        "num_serialized_models_to_keep": 10,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": 0,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        }
    }
}

from allennlp.common import Params
import os
from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.dataset import Batch
from allennlp.nn import util
import torch
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.data import Vocabulary


@TokenEmbedder.register("srl_embedder")
class SRLEmbedder(TokenEmbedder):
    def __init__(self,
                 serialization_dir,
                 cuda_device=0) -> None:
        super(SRLEmbedder, self).__init__()

        from allennlp.models.archival import load_archive

        self.serialization_dir = serialization_dir
        self.parameter_filename = os.path.join(serialization_dir, "config.json")
        self.weights_filename = os.path.join(serialization_dir, "weights.th")
        self.cuda_device = cuda_device

        self.config = Params.from_file(self.parameter_filename)
        self.archive = load_archive(self.serialization_dir)
        self.model = self.archive.model
        self.model.eval()
        self.dataset_reader_params = self.config["dataset_reader"]
        self.dataset_reader = DatasetReader.from_params(self.dataset_reader_params)
        self.tokenizer = SpacyWordSplitter(language='en_core_web_sm', ner=True, wst=True)

    def forward(self, inputs):
        texts = self.inputs_to_texts(inputs)
        instances = self.texts_to_instances(texts)
        dataset = Batch(instances)
        dataset.index_instances(self.model.vocab)
        srl_inputs = util.move_to_device(dataset.as_tensor_dict(), self.cuda_device)
        tokens = srl_inputs['tokens']
        verb_indicator = srl_inputs['verb_indicator']
        embedded_text_input = self.model.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)
        embedded_verb_indicator = self.model.binary_feature_embedding(verb_indicator.long())
        embedded_text_with_verb_indicator = torch.cat([embedded_text_input, embedded_verb_indicator], -1)
        encoded_text = self.model.encoder(embedded_text_with_verb_indicator, mask)

        return encoded_text.detach()

    def texts_to_instances(self, texts):
        instances = []
        for text in texts:
            tokens = self.tokenizer.split_words(text)
            verb_labels = [0 for _ in tokens]
            instance = self.dataset_reader.text_to_instance(tokens, verb_labels)
            instances.append(instance)
        return instances

    def inputs_to_texts(self, inputs, k='words'):
        texts = [' '.join(x[k]) for x in inputs['metadata']]
        return texts

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params):
        serialization_dir = params.pop('serialization_dir')
        cuda_device = params.pop_int('cuda_device')
        return cls(serialization_dir, cuda_device)

    def get_output_dim(self) -> int:
        return self.model.encoder.get_output_dim()




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
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn.util import remove_sentence_boundaries, add_sentence_boundary_token_ids, get_device_of
from torch.nn.modules import Dropout


@TokenEmbedder.register("ner_embedder_2")
class NEREmbedding2(TokenEmbedder):
    def __init__(self,
                 serialization_dir,
                 cuda_device=0) -> None:
        super(NEREmbedding2, self).__init__()
        
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
        
        num_output_representations = 1
        do_layer_norm = False
        num_layers = 3
        self._keep_sentence_boundaries = False
        self._dropout = Dropout(p=0.5)
        
        self._scalar_mixes: Any = []
        for k in range(num_output_representations):
            scalar_mix = ScalarMix(num_layers, do_layer_norm=do_layer_norm)
            self.add_module('scalar_mix_{}'.format(k), scalar_mix)
            self._scalar_mixes.append(scalar_mix)
    

    def forward(self, inputs, elmo_lstm_output):
        texts = self.inputs_to_texts(inputs)
        instances = self.texts_to_instances(texts)
        dataset = Batch(instances)
        dataset.index_instances(self.model.vocab)
        cp_inputs = util.move_to_device(dataset.as_tensor_dict(), self.cuda_device)
        tokens = cp_inputs['tokens']
        mask = get_text_field_mask(tokens)
        
        layer_activations = elmo_lstm_output['activations']
        mask_with_bos_eos = elmo_lstm_output['mask']

        # compute the elmo representations
        representations = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
            representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
            if self._keep_sentence_boundaries:
                processed_representation = representation_with_bos_eos
                processed_mask = mask_with_bos_eos
            else:
                representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(
                        representation_with_bos_eos, mask_with_bos_eos)
                processed_representation = representation_without_bos_eos
                processed_mask = mask_without_bos_eos
            representations.append(self._dropout(processed_representation))

        # reshape if necessary
        mask = processed_mask
        elmo_representations = representations
            
        embedded_text_input = elmo_representations[0]
        
        encoded_text = self.model.encoder(embedded_text_input, mask)
        return encoded_text.detach()

    def texts_to_instances(self, texts):
        instances = []
        for text in texts:
            tokens = self.tokenizer.split_words(text)
            instance = self.dataset_reader.text_to_instance(tokens)
            instances.append(instance)
        return instances

    def inputs_to_texts(self, inputs, k = 'words'):
        texts = [' '.join(x[k]) for x in inputs['metadata']]
        return texts

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params):
        serialization_dir = params.pop('serialization_dir')
        cuda_device = params.pop_int('cuda_device')
        return cls(serialization_dir, cuda_device)

    def get_output_dim(self) -> int:
        return self.model.encoder.get_output_dim()

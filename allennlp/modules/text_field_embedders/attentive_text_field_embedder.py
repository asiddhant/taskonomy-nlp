import torch.nn as nn
from typing import Dict, List
import warnings
import torch
from overrides import overrides
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders import DependencyParsingEmbedding, ConstituencyParsingEmbedding, NEREmbedding
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.seq2seq_encoders import MultiHeadSelfAttention,PytorchSeq2SeqWrapper
import allennlp.nn.util as util
from torch.nn import LSTM, GRU

import os

@TextFieldEmbedder.register("attention")
class AttentiveTextFieldEmbedder(TextFieldEmbedder):
    """
    This is a ``TextFieldEmbedder`` that wraps a collection of :class:`TokenEmbedder` objects.  Each
    ``TokenEmbedder`` embeds or encodes the representation output from one
    :class:`~allennlp.data.TokenIndexer`.  As the data produced by a
    :class:`~allennlp.data.fields.TextField` is a dictionary mapping names to these
    representations, we take ``TokenEmbedders`` with corresponding names.  Each ``TokenEmbedders``
    embeds its input, and the result is concatenated in an arbitrary order.
    Parameters
    ----------
    token_embedders : ``Dict[str, TokenEmbedder]``, required.
        A dictionary mapping token embedder names to implementations.
        These names should match the corresponding indexer used to generate
        the tensor passed to the TokenEmbedder.
    allow_unmatched_keys : ``bool``, optional (default = False)
        If True, then don't enforce the keys of the ``text_field_input`` to
        match those in ``token_embedders`` (useful if the mapping is specified
        via ``embedder_to_indexer_map``).
    """
    def __init__(self,
                 token_embedders: Dict[str, TokenEmbedder],
                 output_dim: int,
                 allow_unmatched_keys: bool = False) -> None:
        super(AttentiveTextFieldEmbedder, self).__init__()
        self._token_embedders = token_embedders
        self.output_dim = output_dim

        for key, embedder in token_embedders.items():
            name = 'token_embedder_%s' % key
            self.add_module(name, embedder)
        self._allow_unmatched_keys = allow_unmatched_keys

        self.use_glove = False
        if 'tokens' in self._token_embedders :
            self.use_glove = True
            self.glove_embedder = self._token_embedders['tokens']

        self.use_elmo = False
        if 'elmo' in self._token_embedders :
            self.use_elmo = True
            self.elmo_embedder = self._token_embedders['elmo']

        self.use_char = False
        if 'token_characters' in self._token_embedders:
            self.use_char = True
            self.char_embedder = self._token_embedders['token_characters']

        self.num_tasks = len(self._token_embedders) - int(self.use_glove) - int(self.use_elmo) - int(self.use_char)

        self.separate_embedder_keys = set(['tokens', 'elmo', 'token_characters'])
        self.linear_layers = {}
        for key, embedder in self._token_embedders.items():
            if key in self.separate_embedder_keys:
                continue
            in_dim = embedder.get_output_dim()
            out_dim = self.output_dim
            self.linear_layers[key] = nn.Linear(in_dim, out_dim, bias=False)
            if torch.cuda.is_available():
                self.linear_layers[key].cuda()

        seaparate_embedding_total_dim = 0
        for key in self.separate_embedder_keys:
            if key in self._token_embedders:
                embedder = self._token_embedders[key]
                seaparate_embedding_total_dim += embedder.get_output_dim()

        self.lstm = lstm = LSTM(bidirectional=True, num_layers=1, input_size=output_dim,
                                hidden_size=output_dim, batch_first=True)
        self.rnn_encoder = PytorchSeq2SeqWrapper(lstm)
        self.attention = MultiHeadSelfAttention(num_heads=4,
                                                input_dim=output_dim,
                                                attention_dim=output_dim,
                                                values_dim=output_dim,
                                                output_projection_dim=output_dim)

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim + (self.glove_embedder.get_output_dim() if self.use_glove else 0) + \
                (self.elmo_embedder.get_output_dim() if self.use_elmo else 0) + \
               (self.char_embedder.get_output_dim() if self.use_char else 0)

    def forward(self, tokens, num_wrapping_dims: int = 0) -> torch.Tensor:
        embedded_representations = []
        keys = sorted(self._token_embedders.keys())
        for key in keys:
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            if key in self.separate_embedder_keys:
                continue
            embedder = getattr(self, 'token_embedder_{}'.format(key))
            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)
            token_vectors = self.linear_layers[key](embedder(tokens))
            embedded_representations.append(token_vectors)

        mask = util.get_text_field_mask(tokens)
        embedded_representations = torch.stack(embedded_representations,dim=-2)
        query_emb = self.rnn_encoder(tokens,mask)
        similarities = torch.matmul(embedded_representations,query_emb.unsqueeze(-1)).squeeze(-1)
        similarities = util.masked_softmax(similarities,mask,dim=-1)
        combined_emb = torch.matmul(embedded_representations.transpose(2,3), similarities.unsqueeze(-1)).squeeze(-1)

        if self.use_glove :
            embedder = getattr(self, 'token_embedder_tokens')
            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(self.glove_embedder)
            glove_emb = embedder(tokens['tokens'])
            combined_emb = torch.cat([combined_emb, glove_emb],dim=-1)

        if self.use_elmo :
            embedder = getattr(self, 'token_embedder_elmo')
            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(self.elmo_embedder)
            elmo_emb = embedder(tokens['elmo'])
            combined_emb = torch.cat([combined_emb, elmo_emb],dim=-1)

        if self.use_char:
            embedder = getattr(self, 'token_embedder_token_characters')
            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(self.char_embeddder)
            token_vectors = embedder(tokens['token_characters'])
            combined_emb = torch.cat([combined_emb, token_vectors], dim=-1)

        return combined_emb

    # This is some unusual logic, it needs a custom from_params.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'WeightedAverageTextFieldEmbedder':  # type: ignore
        # pylint: disable=arguments-differ,bad-super-call

        # The original `from_params` for this class was designed in a way that didn't agree
        # with the constructor. The constructor wants a 'token_embedders' parameter that is a
        # `Dict[str, TokenEmbedder]`, but the original `from_params` implementation expected those
        # key-value pairs to be top-level in the params object.
        #
        # This breaks our 'configuration wizard' and configuration checks. Hence, going forward,
        # the params need a 'token_embedders' key so that they line up with what the constructor wants.
        # For now, the old behavior is still supported, but produces a DeprecationWarning.

        allow_unmatched_keys = params.pop_bool("allow_unmatched_keys", False)

        token_embedder_params = params.pop('token_embedders', None)
        output_dim = params.pop_int('output_dim')

        if token_embedder_params is not None:
            # New way: explicitly specified, so use it.
            token_embedders = {
                    name: TokenEmbedder.from_params(subparams, vocab=vocab)
                    for name, subparams in token_embedder_params.items()
            }

        else:
            # Warn that the original behavior is deprecated
            warnings.warn(DeprecationWarning("the token embedders for WeightedAverageTextEmbedding should now "
                                             "be specified as a dict under the 'token_embedders' key, "
                                             "not as top-level key-value pairs"))

            token_embedders = {}
            keys = list(params.keys())
            for key in keys:
                embedder_params = params.pop(key)
                token_embedders[key] = TokenEmbedder.from_params(vocab=vocab, params=embedder_params)

        params.assert_empty(cls.__name__)
        return cls(token_embedders, output_dim, allow_unmatched_keys)

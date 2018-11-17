"""
A :class:`~allennlp.modules.token_embedders.token_embedder.TokenEmbedder` is a ``Module`` that
embeds one-hot-encoded tokens as vectors.
"""

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.modules.token_embedders.elmo_token_embedder_2 import ElmoTokenEmbedder2
from allennlp.modules.token_embedders.openai_transformer_embedder import OpenaiTransformerEmbedder
from allennlp.modules.token_embedders.dependency_embedder import DependencyParsingEmbedding
from allennlp.modules.token_embedders.srl_embedder import SRLEmbedding
from allennlp.modules.token_embedders.dependency_embedder_2 import DependencyParsingEmbedding2
from allennlp.modules.token_embedders.constituency_embedder import ConstituencyParsingEmbedding
from allennlp.modules.token_embedders.constituency_embedder_2 import ConstituencyParsingEmbedding2
from allennlp.modules.token_embedders.ner_embedder import NEREmbedding
from allennlp.modules.token_embedders.ner_embedder_2 import NEREmbedding2

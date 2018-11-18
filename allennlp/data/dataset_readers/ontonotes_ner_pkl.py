import logging
from typing import Dict, List, Iterable

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence, to_bioul


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def _normalize_word(word: str):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

@DatasetReader.register("ontonotes_ner_pkl")
class OntonotesNamedEntityRecognitionPkl(DatasetReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for fine-grained named entity recognition. It returns a dataset of instances with the
    following fields:

    tokens : ``TextField``
        The tokens in the sentence.
    tags : ``SequenceLabelField``
        A sequence of BIO tags for the NER classes.

    Note that the "/pt/" directory of the Onotonotes dataset representing annotations
    on the new and old testaments of the Bible are excluded, because they do not contain
    NER annotations.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    domain_identifier: ``str``, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.
    coding_scheme : ``str``, (default = None).
        The coding scheme to use for the NER labels. Valid options are "BIO" or "BIOUL".

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Fine-Grained NER.

    """
    @overrides
    def _read(self, file_path: str):
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        load_data = pkl.load(open(file_path,'rb'))
        logger.info("Number of Instances Found: %s", len(load_data))
        for x in load_data:
            yield x
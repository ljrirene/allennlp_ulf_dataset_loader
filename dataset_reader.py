from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, SpacyTokenizer
from typing import Dict, Iterable, List, Iterator
import json
import logging

logger = logging.getLogger(__name__)


@DatasetReader.register("ulf_reader")
class ULFReader(DatasetReader):
    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, sentence: str, ulf: str, ulf_amr: str) -> Instance:
        tokens = self.tokenizer.tokenize(sentence)
        text_field = TextField(tokens, self.token_indexers)
        fields: Dict[str, Field] = {"text": text_field}
        if ulf:
            fields["ulf"] = LabelField(ulf)
        if ulf_amr:
            fields["ulf_amr"] = LabelField(ulf_amr)
        return Instance(fields)

    def _read(self, file_path):
        print("in")
        with open(file_path) as f:
            lines = json.load(f)
        logging.info("Reading entries from lines in file at %s", file_path)
        for line in lines:
            _, sentence, ulf, ulf_amr = line[0], line[1], line[2], line[3]
            yield self.text_to_instance(sentence, ulf, ulf_amr)


if __name__ == '__main__':
    reader = ULFReader()
    reader.read("./ulf-data-1.0/ulf/dev/all.json")

from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
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
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path):
        with open(file_path) as f:
            lines = json.load(f)
        logging.info("Reading entries from lines in file at %s", file_path)
        for line in lines:
            _, sentence, ulf, ulf_amr = line[0], line[1], line[2], line[3]
            yield self.text_to_instance(sentence, ulf, ulf_amr)

    def text_to_instance(self, sentence: str, ulf: str, ulf_amr: str) -> Instance:
        tokens = self.tokenizer.tokenize(sentence)
        text_field = TextField(tokens, self.token_indexers)
        fields: Dict[str, Field] = {"text": text_field}
        if ulf:
            fields["ulf"] = LabelField(ulf)
        if ulf_amr:
            fields["ulf_amr"] = LabelField(ulf_amr)
        return Instance(fields)


if __name__ == '__main__':
    dataset_reader = ULFReader()
    instances = list(dataset_reader.read("./ulf-data-1.0/ulf/dev/all.json"))

    for instance in instances[:1]:
        print(instance)

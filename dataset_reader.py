from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from typing import Dict, Iterable, List, Iterator
import csv
import json
import logging
logger = logging.getLogger(__name__)


@DatasetReader.register("ulf_reader")
class ULFReader(DatasetReader):
    def __init__(
            self,
            # lazy: bool = False,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None) -> None:
        # super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as dataset_file:
            lines = json.load(dataset_file)
        logging.info("Reading entries from lines in file at %s", file_path)
        for line in lines:
            _, sentence, ulf, ulf_amr = line[0], line[1], line[2], line[3]
            yield self.text_to_instance(sentence, ulf, ulf_amr)

    def text_to_instance(self, sentence: str, ulf: str, ulf_amr: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(sentence)
        text_field = TextField(tokens, self.token_indexers)
        fields: Dict[str, Field] = {"text": text_field}
        if ulf:
            fields["ulf"] = LabelField(ulf)
        if ulf_amr:
            fields["ulf_amr"] = LabelField(ulf_amr)
        return Instance(fields)


if __name__ == '__main__':
    print("test")

    dataset_reader = ULFReader()
    dataset_reader.read("test.json")


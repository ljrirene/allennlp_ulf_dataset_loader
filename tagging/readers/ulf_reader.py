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
            **kwargs) -> None:
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
        tokenized_ulf = self.tokenizer.tokenize(ulf)
        tokenized_ulf_amr = self.tokenizer.tokenize(ulf_amr)
        ulf_field = TextField(tokenized_ulf, self.token_indexers)
        ulf_amr_field = TextField(tokenized_ulf_amr, self.token_indexers)

        fields: Dict[str, Field] = {"text": text_field, "ulf": ulf_field, "ulf_amr": ulf_amr_field}

        return Instance(fields)



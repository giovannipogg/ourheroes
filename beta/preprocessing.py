import asyncio
from dataclasses import dataclass
from typing import Union, Optional, Sequence

import torch
import torchtext.vocab

from ourtypes import Tokenized, ProcessedString, Stage, Token


@dataclass
class Remover:
    removes: Sequence[str] = ('<S>', '</S>', '\n')

    def remove(self, sentence: str):
        for remove in self.removes:
            sentence = sentence.replace(remove, ' ')
        return sentence

    def __call__(self, sentence: str):
        return self.remove(sentence)

    def __str__(self):
        return f"Remover(removes={self.removes})"


def lowerize(sentence: str) -> str:
    return sentence.lower()


@dataclass
class SimpleTokenizer:
    max_tokens: Optional[int] = 128

    def tokenize(self, sentence: str) -> Tokenized:
        sentence += '!'  # grant all tokens to output
        tokens = []
        current_token = ''
        for c in sentence:
            if c.isalnum():
                current_token += c
            else:
                if current_token != '':
                    tokens.append(current_token)
                if not c.isspace():
                    tokens.append(c)
                current_token = ''
        n = len(tokens) - 1
        if self.max_tokens is not None:
            n = min([n, self.max_tokens])
        return tokens[:n]  # exclude spurious '!'

    def __call__(self, sentence: str) -> Tokenized:
        return self.tokenize(sentence)


# class GlobalGlove:
#     __glove: Optional[torchtext.vocab.GloVe] = None
#
#     def __call__(cls):
#         if cls.__glove is None:
#             cls.__glove = torchtext.vocab.GloVe()
#         return cls.__glove
#
#     def __getitem__(cls, key: Token) -> torch.Tensor:
#         if cls.__glove is None:
#             cls()
#         return cls.__glove[key]


@dataclass
class GloVeTensorizer:
    vectors: torchtext.vocab.GloVe = torchtext.vocab.GloVe()
    start_of_sentence: str = '<s>'
    end_of_sentence: str = '</s>'

    def tensorize(self, tokenized: Tokenized) -> torch.Tensor:
        tensors = [self.vectors[self.start_of_sentence]]
        tensors += [self.vectors[token] for token in tokenized]
        tensors += [self.vectors[self.end_of_sentence]]
        return torch.stack(tensors)

    def __call__(self, tokenized: Tokenized) -> torch.Tensor:
        return self.tensorize(tokenized)


class GloVeWrapper:
    vectors: torchtext.vocab.GloVe = torchtext.vocab.GloVe()

    def __call__(self, token: Token) -> torch.Tensor:
        return self.vectors[token]

@dataclass
class PreprocessingPipeline:
    stages: Sequence

    def preprocess(self, sentence: str) -> ProcessedString:
        output = sentence
        for stage in self.stages:
            output = stage(output)
        return output

    def __call__(self, sentence: str) -> ProcessedString:
        return self.preprocess(sentence)


@dataclass
class Recursor:
    process: Union[Stage, PreprocessingPipeline]

    def recurse(self, input: Union[Sequence, str]) -> Union[Sequence, ProcessedString]:
        if isinstance(input, str):
            return self.process(input)
        return type(input)([self.recurse(sub) for sub in input])

    def __call__(self, input: Sequence) -> Sequence:
        return self.recurse(input)


@dataclass
class AsyncRecursor:
    process: Union[Stage, PreprocessingPipeline]

    async def recurse(self, input: Union[Sequence, str]) -> Union[Sequence, ProcessedString]:
        if isinstance(input, str):
            return self.process(input)
        output = await asyncio.gather(*[self.recurse(sub) for sub in input])
        return type(input)(output)

    async def __call__(self, input: Sequence) -> Sequence:
        output = await self.recurse(input)
        return output


"""Module implementing the tokenizer classes.

These classes are used to break up sentences into token vectors.
"""

from dataclasses import dataclass
from typing import Optional

from ourheroes.data.types import Tokenized, Tokenizer, Vocab


@dataclass
class SimpleTokenizer:
    max_tokens: Optional[int] = None

    """A minimal tokenizer.
    
    Since the Arxiv and PubMed datasets already come
    with tokenized sentences, this class further
    separates composite words in order to gain in
    representability.
    
    Attributes:
        max_tokens: The maximum number of tokens
            retained when the object is called,
            after which a sentence is truncated.
    
    Example:
        Words like 'l-arginine' may not find
        a representation within GloVe's learned
        embeddings while the sub-words 'l', '-'
        and 'arginine' do.
    """

    def tokenize(self, sentence: str) -> Tokenized:
        """Splits the sentence into tokens.
        
        Args:
            sentence: The sentence to tokenize.
        Returns:
            The tokenized sentence up to max_tokens.
        """
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
        """Splits the sentence into tokens,
        equivalent to self.tokenize(sentence).
        
        Args:
            sentence: The sentence to tokenize.
        Returns:
            The tokenized sentence up to max_tokens.
        """
        return self.tokenize(sentence)


@dataclass
class NLTKTokenizer:
    tokenizer: Tokenizer
    vocab: Optional[Vocab] = None
    max_tokens: Optional[int] = None

    def tokenize(self, sentence: str) -> Tokenized:
        sentence = self.tokenizer(sentence)
        if self.vocab is not None:
            sentence = [token for token in sentence if token in self.vocab]
        if self.max_tokens is not None:
            return sentence[:self.max_tokens]
        return sentence

    def __call__(self, sentence: str) -> Tokenized:
        return self.tokenize(sentence)

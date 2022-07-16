import asyncio
import json
import os
from collections import defaultdict
from pprint import pprint
from typing import Sequence, Tuple, List

from tqdm import tqdm

from preprocessing import SimpleTokenizer
import torch
import torchtext.vocab


async def file_tokens(file: str, tq: tqdm) -> Sequence[Tuple[str, int, str]]:
    with open(file, 'r', encoding='utf-8') as fp:
        file = json.load(fp)
    tokens = defaultdict(lambda: 0)
    for sentence in file['article_text']:
        for token in sentence.split(' '):
            tokens[token.lower()] += 1
    tq.update(1)
    return tokens

async def main(files: List[str]):
    save_path = 'tokens_infos.json'

    if not os.path.isfile(save_path):
        tq = tqdm(total=len(files))
        result = await asyncio.gather(*[file_tokens(file, tq) for file in files])
        tq.close()
        tokens_count = defaultdict(lambda: 0)
        for sub in result:
            for token, value in sub.items():
                tokens_count[token] += value
        tokens_count = dict(tokens_count)
        with open(save_path, 'w') as fp:
            json.dump(tokens_count, fp)
            fp.flush()
    else:
        with open(save_path, 'r') as fp:
            tokens_count = json.load(fp)

    print("=============================")
    print(f"Total tokens: {len(tokens_count)}")
    print("=============================")

    vectorizer = torchtext.vocab.GloVe()
    tokenizer = SimpleTokenizer()
    zero_tensor = torch.Tensor([0.])

    nonrepresentable = []
    tq = tqdm(total=len(tokens_count))
    for TOKEN in tokens_count.keys():
        tokens = tokenizer(TOKEN)
        fully_representable = True
        for token in tokens:
            if torch.allclose(vectorizer[token], zero_tensor):
                fully_representable = False
        if not fully_representable:
            nonrepresentable.append(TOKEN)
        tq.update(1)
    tq.close()

    with open('nonrepresentable.json', 'w') as fp:
            json.dump(nonrepresentable, fp)
            fp.flush()

    exit()

    representable = set()
    non_representable = set()
    # tokenized_nr = set()

    print("=============================")
    print(f"Repr tokens: {len(representable)}")
    print("=============================")

    repr_after_tok = set()
    part_after_tok = set()
    non_after_tok = set()
    tq = tqdm(total=len(non_representable))
    for token in non_representable:
        tokenized = tokenizer(token)
        all_sub = True
        part = False
        for sub in tokenized:
            if torch.allclose(vectorizer[sub], zero_tensor):
                all_sub = False
            else:
                part = True
        if all_sub:
            repr_after_tok.add(token)
        if not all_sub and part:
            part_after_tok.add(token)
        if not all_sub and not part:
            non_after_tok.add(token)
        tq.update(1)
    tq.close()

    print("=============================")
    print(f"Repr after: {len(repr_after_tok)}")
    print("=============================")
    print("=============================")
    print(f"Part after: {len(part_after_tok)}")
    print("=============================")
    print(f"Non after: {len(non_after_tok)}")

    # print("=============================")
    # print(f"NRep tokens: {len(non_representable)}")
    # print("=============================")
    # print("=============================")
    # print(f"NR tokenized: {len(tokenized_nr)}")
    # print("=============================")
    # print("=============================")
    # print(f"NRT inter Repr: {len(tokenized_nr.intersection(representable))}")
    # print("=============================")
    #
    # pprint(tokenized_nr.intersection(representable))
    # pprint(tokenized_nr.difference(representable))



if __name__ == '__main__':
    root = 'C:\\Users\\giova\\Documents\\Poli\\S3\\NLP\\Project\\pubmed-dataset\\parsed\\train'
    files = [f'{root}\\{file}' for file in os.listdir(root) if file.endswith('.txt')]
    asyncio.run(main(files))
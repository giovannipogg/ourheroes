import json
import os

from dataset import DocumentDataset, CRDataset, GSDataset
from ourheroes import OurHeroes


def data_loading_test():
    with open('retained.json', 'r') as fp:
        files = json.load(fp)

    doc_dataset = DocumentDataset(files)
    # cr_dataset = CRDataset(doc_dataset)

    # val = cr_dataset[0]
    # print(val)

    return doc_dataset


def cr_test():
    model = OurHeroes()
    doc_dataset = data_loading_test()
    model.fit(doc_dataset)

def digest_path(file: str) -> str:
    file = file.split('\\')
    root, article = '\\'.join(file[:-1]), file[-1]
    return f"{root}\\digests\\{article}"

def gs_dataset_test():
    with open('retained.json', 'r') as fp:
        files = json.load(fp)
    files = [(file, digest_path(file)) for file in files]
    files = [file for file in files if os.path.isfile(file[1])]
    gs_dataset = GSDataset(files)
    val = gs_dataset[0]
    print()

from nltk.tokenize import T

if __name__ == '__main__':
    # gs_dataset_test()

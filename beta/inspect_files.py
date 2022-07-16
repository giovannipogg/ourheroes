import asyncio
import json
import os
import string
from collections import defaultdict
from typing import Tuple, Dict, Callable, Optional, Iterable
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from tqdm import tqdm

from preprocessing import SimpleTokenizer, Recursor, PreprocessingPipeline, Remover, lowerize, ProcessedString

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

FIGSIZE = WIDTH, HEIGHT = 12, 8
DEFAULT_QS = (.9, .95, .97, .99, .995, .997, .999)

def plot_line(stats: np.ndarray, qs: Iterable[float], title: str, ax: Optional[Axes] = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    ys, xs = np.unique(stats, return_counts=True)
    xs = np.cumsum(xs)
    xs = xs / xs[-1]
    vs = np.quantile(stats, qs)
    if xs.size < 20:
        ax.plot(xs, ys, linewidth=2, color='black', marker='s')
    else:
        ax.plot(xs, ys, linewidth=2, color='black')
    kwargs = {'linewidth': 1, 'linestyle': '--'}
    for q, v in zip(qs, vs):
        if q > .5:
            kwargs['color'] = 'blue'
        else:
            kwargs['color'] = 'red'
        ax.axhline(v, **kwargs)
        ax.axvline(q, **kwargs)
        ax.plot([], [], label=f'{q*100:.2f} % = {v:.2f}', **kwargs)
    ax.legend()
    ax.grid()
    ax.set_title(title)
    return ax

def plot_hist(stats: np.ndarray, qs: Iterable[float], title: str, ax: Optional[Axes] = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    vs = np.quantile(stats, qs)
    n = min([100, len(np.unique(stats))])
    ax.hist(stats, density=True, bins=n)
    kwargs = {'linewidth': 1, 'linestyle': '--'}
    for q, v in zip(qs, vs):
        if q > .5:
            kwargs['color'] = 'blue'
        else:
            kwargs['color'] = 'red'
        ax.axvline(v, label=f'{q*100:.2f} % = {v:.2f}', **kwargs)
    ax.legend()
    ax.grid()
    ax.set_title(title)
    return ax

async def get_stats(file: str, tq: tqdm, process: Callable[[str], ProcessedString]) -> Tuple[str, Dict]:
    tq.update(1)
    with open(file, 'r', encoding='utf-8') as fp:
        doc = json.load(fp)
    sections = doc['sections']
    summary = doc['abstract_text']
    statistics = {
        'number_of_sections': len(sections),
        'number_of_sentences_section': [len(section) for section in sections],
        'length_sentences_section': [[len(process(sentence)) for sentence in section] for section in sections],
        'number_of_sentences_summary': len(summary),
        'length_sentences_summary': [len(process(sentence)) for sentence in summary]
    }
    return file, statistics


async def main():
    root = 'C:\\Users\\giova\\Documents\\Poli\\S3\\NLP\\Project\\pubmed-dataset\\parsed\\val'
    files = [f"{root}\\{file}" for file in os.listdir(root) if file.endswith('.txt')]
    stats_file = 'stats_val.json'

    with open('nonrepresentable.json', 'r') as fp:
        nonrepresentable = json.load(fp)

    gs_non_represented = nonrepresentable + list(string.punctuation) + list(stopwords.words('english'))
    process = Recursor(
        PreprocessingPipeline(
            [Remover(), lowerize, Remover(gs_non_represented), SimpleTokenizer()]
        )
    )


    if os.path.isfile(stats_file):
        with open(stats_file, 'r') as fp:
            statistics = json.load(fp)
    else:
        tq = tqdm(total=len(files))
        statistics = await asyncio.gather(*[get_stats(file, tq, process) for file in files])
        tq.close()
        statistics = {file:stats for file, stats in statistics}
        with open(stats_file, 'w') as fp:
            json.dump(statistics, fp, indent=4)
            fp.flush()

    exit()

    file_statistics = {}
    document_statistics = defaultdict(lambda: list())
    section_statistics = defaultdict(lambda: list())
    sentence_statistics = defaultdict(lambda: list())
    for k, v in statistics.items():
        tmp = {}

        document_statistics['number_of_sections'].append(v['number_of_sections'])
        tmp['number_of_sections'] = v['number_of_sections']

        values = v['number_of_sentences_section']
        document_statistics['tot_sentences'].append(sum(values))
        tmp['tot_sentences'] = sum(values)

        values = [n_tokens for section in v['length_sentences_section'] for n_tokens in section]
        document_statistics['tot_tokens'].append(sum(values))
        tmp['tot_tokens'] = sum(values)

        document_statistics['tot_sentences_summary'].append(v['number_of_sentences_summary'])
        tmp['tot_sentences_summary'] = v['number_of_sentences_summary']
        values = v['length_sentences_summary']
        document_statistics['tot_tokens_summary'].append(sum(values))
        tmp['tot_tokens_summary'] = sum(values)

        section_statistics['number_of_sentences'] += v['number_of_sentences_section']

        values = v['length_sentences_section']
        section_statistics['tot_tokens'] += [sum(value) for value in values]

        sentence_statistics['n_tokens'] += [n_tokens for section in v['length_sentences_section'] for n_tokens in section]

        file_statistics[k] = tmp

    qs = [.001, .003, .005, .95, .97, .99, .995, .997, .999]
    for k, v in document_statistics.items():
        plot_line(np.array(v), qs, f"doc {k}")
        plt.show()

    for k, v in section_statistics.items():
        plot_line(np.array(v), qs, f"sec {k}")
        plt.show()

    for k, v in sentence_statistics.items():
        plot_line(np.array(v), qs, f"sen {k}")
        plt.show()

    MIN_SECTIONS = 1
    MAX_SECTIONS = 16
    MIN_TOT_SENTENCES = 16
    MAX_TOT_SENTENCES = 320
    MIN_TOT_TOKENS = 400
    MAX_TOT_TOKENS = 12_000
    MIN_TOT_SENTENCES_SUMMARY = 2
    MAX_TOT_SENTENCES_SUMMARY = 16
    MIN_TOT_TOKENS_SUMMARY = 50
    MAX_TOT_TOKENS_SUMMARY = 500

    MIN_SENTENCES_SECTION = 1
    MAX_SENTENCES_SECTION = 100
    MIN_TOKENS_SECTION = 1
    MAX_TOKENS_SECTION = 3_200

    MIN_TOKENS_SENTENCE = 1
    MAX_TOKENS_SENTENCE = 216

    # SELECTIONS
    retain = defaultdict(lambda: True)
    for f, s in file_statistics.items():
        # First selection: ill-formed documents
        # exclude documents without body or summary, or with a summary longer than the body
        if s['tot_tokens'] == 0 or s['tot_tokens_summary'] == 0 or s['tot_tokens_summary'] > s['tot_tokens']:
            retain[f] = False
        # document level selections
        if s['number_of_sections'] < MIN_SECTIONS or s['number_of_sections'] > MAX_SECTIONS:
            retain[f] = False
        if s['tot_sentences'] < MIN_TOT_SENTENCES or s['tot_sentences'] > MAX_TOT_SENTENCES:
            retain[f] = False
        if s['tot_tokens'] < MIN_TOT_TOKENS or s['tot_tokens'] > MAX_TOT_TOKENS:
            retain[f] = False
        if s['tot_sentences_summary'] < MIN_TOT_SENTENCES_SUMMARY or s['tot_sentences_summary'] > MAX_TOT_SENTENCES_SUMMARY:
            retain[f] = False
        if s['tot_tokens_summary'] < MIN_TOT_TOKENS_SUMMARY or s['tot_tokens_summary'] > MAX_TOT_TOKENS_SUMMARY:
            retain[f] = False
        val = statistics[f]['number_of_sentences_section']
        if any([v < MIN_SENTENCES_SECTION for v in val]) or any([v > MAX_SENTENCES_SECTION for v in val]):
            retain[f] = False
        val = [sum(n_tokens) for n_tokens in statistics[f]['length_sentences_section']]
        if any([v < MIN_TOKENS_SECTION for v in val]) or any([v > MAX_TOKENS_SECTION for v in val]):
            retain[f] = False
        val = [n_tokens for section in statistics[f]['length_sentences_section'] for n_tokens in section]
        if any([v < MIN_TOKENS_SENTENCE for v in val]) or any([v > MAX_TOKENS_SENTENCE for v in val]):
            retain[f] = False

    filtered = [f for f in statistics.keys() if retain[f]]
    print(len(filtered), len(filtered)/len(statistics))

    with open('retained.json', 'w') as fp:
        json.dump(filtered, fp)
        fp.flush()
    exit()



    file_statistics2 = {}
    file_statistics_statistics = defaultdict(lambda: list())
    for f, d in file_statistics.items():
        if not retain[f]:
            continue
        for k, v in d.items():
            file_statistics_statistics[k].append(v)
        file_statistics2[f] = d
    file_statistics = file_statistics2

    for k, v in file_statistics_statistics.items():
        plot_line(np.array(v), [.005, .01, .03, .95, .97, .99], k)
        plt.show()

    for k, v in globals.items():
        plot_line(np.array(v), [.005, .01, .03, .95, .97, .99], k)
        plt.show()

    exit()

    # Second selection: memory limits
    # exclude documents with lenghty bodies
    fig, ax = plt.subplots(figsize=(2*WIDTH, 3*HEIGHT), nrows=3, ncols=2)
    stat = np.array(file_statistics_statistics['tot_tokens_body'])
    plot_line(stat, [.95, .97, .99], '# Tokens (body)', ax[0])
    stat = np.array(file_statistics_statistics['tot_sentences_body'])
    plot_line(stat, [.95, .97, .99], '# Sentence (body)', ax[1])
    stat = np.array(file_statistics_statistics['max_sentences_section'])
    plot_line(stat, [.95, .97, .99], 'Max sentences in section (body)', ax[2])
    plt.show()

    MAX_TOKENS = 10_000
    MAX_TOT_SENTENCES = 200
    MAX_SENTENCES_SECTION = 80
    for f, s in file_statistics.items():
        if s['tot_tokens_body'] > MAX_TOKENS or\
                s['tot_sentences_body'] > MAX_TOT_SENTENCES or\
                s['max_sentences_section'] > MAX_SENTENCES_SECTION:
            retain[f] = False

    file_statistics2 = {}
    file_statistics_statistics = defaultdict(lambda: list())
    for f, d in file_statistics.items():
        if not retain[f]:
            continue
        for k, v in d.items():
            file_statistics_statistics[k].append(v)
        file_statistics2[f] = d
    file_statistics = file_statistics2

    # Third selection: ratio outliers
    # exclude documents with abnormal summary/body ratios
    stat1 = np.array(file_statistics_statistics['tot_tokens_body'])
    stat2 = np.array(file_statistics_statistics['tot_tokens_summary'])
    stat = stat2 / stat1
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_line(stat, [.001, .003, .005, .01, .95, .97, .99], '# Tokens (body)', ax)
    plt.show()

    MIN_RATIO = .01
    MAX_RATIO = .25
    for f, s in file_statistics.items():
        ratio = s['tot_tokens_summary'] / s['tot_tokens_body']
        if  MIN_RATIO < ratio < MAX_RATIO:
            continue
        retain[f] = False

    file_statistics2 = {}
    file_statistics_statistics = defaultdict(lambda: list())
    for f, d in file_statistics.items():
        if not retain[f]:
            continue
        for k, v in d.items():
            file_statistics_statistics[k].append(v)
        file_statistics2[f] = d
    file_statistics = file_statistics2

    print(len(file_statistics) / len(statistics))

    # Max tokens:
    # used to set the max_tokens
    stat = np.array(file_statistics_statistics['max_tokens_sentence'])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_line(stat, [.97, .99], 'Maximum tokens in a sentence (body)', ax)
    plt.show()

    retained_files = [file for file in files if retain[file]]
    with open('retained.json', 'w') as fp:
        json.dump(retained_files, fp)
        fp.flush()


if __name__ == '__main__':
    asyncio.run(main())
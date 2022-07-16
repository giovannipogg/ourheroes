# import json
# from dataclasses import dataclass
# from typing import List, Tuple, Callable
#
# Title = Sentence = str
# Summary = Section = List[Sentence]
# Document = List[Tuple[Title, Section]]
#
#
# @dataclass
# class JsonLoader:
#     titles: str = 'section_names'
#     sections: str = 'sections'
#     summary: str = 'abstract_text'
#     encoding: str = 'utf-8'
#
#     def load(self, file_path: str) -> Tuple[Document, Summary]:
#         with open(file_path, 'r', encoding=self.encoding) as fp:
#             document = json.load(fp)
#         titles = document[self.titles]
#         sections = document[self.sections]
#         summary = document[self.summary]
#
#         return [(title, section) for title, section in zip(titles, sections)], summary
#
#     def __call__(self, file_path: str) -> Tuple[Document, Summary]:
#         return self.load(file_path)
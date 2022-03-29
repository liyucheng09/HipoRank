from hipo_rank.dataset_iterators.pubmed import PubmedDataset

from hipo_rank.embedders.sent_transformers import SentTransformersEmbedder

from hipo_rank.similarities.cos import CosSimilarity

from hipo_rank.directions.undirected import Undirected
from hipo_rank.directions.order import OrderBased
from hipo_rank.directions.edge import EdgeBased

from hipo_rank.scorers.add import AddScorer
from hipo_rank.scorers.multiply import MultiplyScorer

from hipo_rank.summarizers.default import DefaultSummarizer

from pathlib import Path
import json
import time
from tqdm import tqdm

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from hipo_rank import Section
from selenium import webdriver
from dataclasses import dataclass
from nltk import sent_tokenize
from typing import List, Dict, Optional, Tuple

@dataclass
class Document:
    # dataclass wrapper for documents yielded by a dataset iterator
    sections: List[Section]
    title: str
    url: str
    meta: Optional[Dict] = None

@dataclass
class Summary:
    title: str
    url: str
    abstract: List[Tuple[str, str]]
    Original_abs: List[str] = None

def read_pdf(file):
    pages = []
    for page_layout in extract_pages(file):
        sentences = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                # print(element.get_text())
                sentences.append(element.get_text())
        text = ''.join(sentences)
        pages.append(text)
    return pages

def load_inference_docs(file_path = 'data/inference'):
    path = Path(file_path)
    # all_files = path.glob('*.pdf')
    all_files = ['data/inference/Making OR practice visible.pdf']
    all_articles = {file: read_pdf(file) for file in all_files}
    docs = []
    for file, article in all_articles.items():
        sections = []
        for page_number, page in enumerate(article):
            sents = page.replace('e.g.', 'e.g.,').replace('al.', 'al,').replace('-', '').replace('\n','').split('. ')
            sents = [i for i in sents if len(i)>20 ]
            sec = Section(page_number, sents)
            sections.append(sec)
        doc = Document(sections, reference=str(file))
        docs.append(doc)
    
    return docs

def output_to_markdown(summ: Summary, path: Path):
    origin_abs = "\n\n".join(summ.Original_abs)
    abses = '\n\n'.join([' --> '.join(sent[::-1]) for sent in summ.abstract])

    output_text = f"""
# {summ.title}

## 原始摘要
{origin_abs}

## 生成摘要
{abses}

"""

    path.write_text(output_text)
    print(f"finished write to {path}")

def get_online_paper(urls):
    """
    urls: a list of sciencedirect pages.
    """

    driver = webdriver.Safari()
    docs = []
    for url in urls:
        driver.get(url)
        time.sleep(3)
        title = driver.find_element_by_class_name('title-text').text
        abstracts = [i.text for i in driver.find_elements_by_xpath('//div[starts-with(@id, "abss")]')]
        secs = []
        for index, sec in enumerate(driver.find_elements_by_xpath('//section[starts-with(@id, "sec")]')):
            sents = []
            sec_title = sec.find_element_by_tag_name('h2').text
            if 'appendix' in sec_title.lower():
                continue
            for para in sec.find_elements_by_tag_name('p'):
                sents.extend(sent_tokenize(para.text))
            secs.append(Section(index, sents, meta={'sec_title': sec_title}))
        doc = Document(secs, title=title, url = url, meta={'abstract': abstracts})
        docs.append(doc)
    
    driver.quit()

    return docs


DATASETS = [
    ("pubmed_test", PubmedDataset, {"file_path": "data/pubmed-release/test.txt"}),
]
EMBEDDERS = [
    ("st_bert_base", SentTransformersEmbedder,
         {"model": "bert-base-nli-mean-tokens"}
        ),
]
SIMILARITIES = [
    ("cos", CosSimilarity, {}),
]
DIRECTIONS = [
    ("edge", EdgeBased, {}),
]

SCORERS = [
    ("add_f=0.0_b=1.0_s=0.5", AddScorer, {"section_weight": 0.5}),
]

urls=['https://www.sciencedirect.com/science/article/pii/S0377221717307476']

def summarize_doc(docs, num_words = 400):
    Summarizer = DefaultSummarizer(num_words=num_words)

    embedder_id, embedder, embedder_args = EMBEDDERS[0]
    Embedder = embedder(**embedder_args)
    embeds = [Embedder.get_embeddings(doc) for doc in tqdm(docs)]
    for similarity_id, similarity, similarity_args in SIMILARITIES:
            Similarity = similarity(**similarity_args)
            print(f"calculating similarities with {similarity_id}")
            sims = [Similarity.get_similarities(e) for e in embeds]
            for direction_id, direction, direction_args in DIRECTIONS:
                print(f"updating directions with {direction_id}")
                Direction = direction(**direction_args)
                sims = [Direction.update_directions(s) for s in sims]
                for scorer_id, scorer, scorer_args in SCORERS:
                    Scorer = scorer(**scorer_args)
                    summaries = []
                    for sim, doc in zip(sims, docs):
                        scores = Scorer.get_scores(sim)
                        summary = Summarizer.get_summary(doc, scores)
                        summ = Summary(doc.title, url=doc.url, abstract=[(sent[0], \
                            doc.sections[sent[2]].meta['sec_title']) for sent in summary], \
                            Original_abs=doc.meta['abstract'])
                        summaries.append(summ)
    return summaries

if __name__ == '__main__':

    results_path = Path(f"data/inference/results")

    docs = get_online_paper(urls)
    summaries = summarize_doc(docs)
    
    for index, summ in enumerate(summaries):
        output_to_markdown(summ, results_path/f'{index}.md')

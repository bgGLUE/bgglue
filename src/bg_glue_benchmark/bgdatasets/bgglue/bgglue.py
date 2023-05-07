import abc
import csv
import dataclasses
import json
import textwrap
from pathlib import Path
from typing import Union

import datasets

_CITATION = """\
@inproceedings{hardalov-etal-2023-bgglue,
    title = "{bgGLUE}: A Bulgarian General Language Understanding Evaluation Benchmark",
    author = "Hardalov, Momchil and 
        Atanasova, Pepa and 
        Mihaylov, Todor and 
        Angelova, Galia and 
        Simov, Kiril and 
        Osenova, Petya and 
        Stoyanov, Ves and 
        Koychev, Ivan and 
        Nakov, Preslav and 
        Radev, Dragomir",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = july,
    year = "2023",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    address = "Toronto, Canada",
    url = "https://arxiv.org/abs/2306.02349"
}
"""

_DESCRIPTION = """\
The Bulgarian General Language Understanding Evaluation (bgGLUE) benchmark is a collection of resources for 
training, evaluating, and analyzing natural language understanding systems in Bulgarian.
"""

_BASE_URL = "https://github.com/bgGLUE/bgglue"
_BASE_DATASET_URL = "https://github.com/bgGLUE/bgglue/raw/main/data"

_CLEF_BASE_URL = (
    "https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/raw/master/task1"
)


@dataclasses.dataclass
class DatasetMeta:
    description: str
    citation: str
    url: str
    data_url: str


_DATASETS_META = {
    "xnlibg": DatasetMeta(
        description=textwrap.dedent(
            """\
          The Cross-lingual Natural Language Inference (XNLI) corpus is a crowd-sourced collection of 5,000 test and
          2,500 dev pairs for the MultiNLI corpus. The pairs are annotated with textual entailment and translated into
          14 languages: French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese,
          Hindi, Swahili and Urdu. This results in 112.5k annotated pairs. Each premise can be associated with the
          corresponding hypothesis in the 15 languages, summing up to more than 1.5M combinations. The corpus is made to
          evaluate how to perform inference in any language (including low-resources ones like Swahili or Urdu) when only
          English NLI data is available at training time. One solution is cross-lingual sentence encoding, for which XNLI
          is an evaluation benchmark."""
        ),
        citation=textwrap.dedent(
            """\
          @InProceedings{conneau2018xnli,
          author = {Conneau, Alexis
                         and Rinott, Ruty
                         and Lample, Guillaume
                         and Williams, Adina
                         and Bowman, Samuel R.
                         and Schwenk, Holger
                         and Stoyanov, Veselin},
          title = {XNLI: Evaluating Cross-lingual Sentence Representations},
          booktitle = {Proceedings of the 2018 Conference on Empirical Methods
                       in Natural Language Processing},
          year = {2018},
          publisher = {Association for Computational Linguistics},
          location = {Brussels, Belgium},
        }"""
        ),
        url="https://www.nyu.edu/projects/bowman/xnli/",
        data_url="https://dl.fbaipublicfiles.com/XNLI",
    ),
    "udep": DatasetMeta(
        description=textwrap.dedent(
            """\
          Universal Dependencies (UD) is a framework for consistent annotation of grammar (parts of speech, morphological
    features, and syntactic dependencies) across different human languages. UD is an open community effort with over 200
    contributors producing more than 100 treebanks in over 70 languages. If you’re new to UD, you should start by reading
    the first part of the Short Introduction and then browsing the annotation guidelines.
    """
        ),
        citation=textwrap.dedent(
            """
        @techreport{OsenovaSimov2004,
            author = {Petya Osenova and Kiril Simov},
            title = {BTB-TR05: BulTreeBank Stylebook ą 05},
            year = {2004},
            url = {http://www.bultreebank.org/TechRep/BTB-TR05.pdf}
        }
        
        @inproceedings{simov-osenova-2003-practical,
            title = "Practical Annotation Scheme for an {HPSG} Treebank of {B}ulgarian",
            author = "Simov, Kiril  and Osenova, Petya",
            booktitle = "Proceedings of 4th International Workshop on Linguistically Interpreted Corpora ({LINC}-03) at {EACL} 2003",
            year = "2003",
            url = "https://aclanthology.org/W03-2403",
        }
        
        @incollection{SimovOsPo2002,
            author = {Kiril Simov and Gergana Popova and Petya Osenova},
            title = {HPSG-based syntactic treebank of Bulgarian (BulTreeBank)},
            booktitle = {A Rainbow of Corpora: Corpus Linguistics and the Languages of the World},
            editor = {Andrew Wilson, Paul Rayson and Tony McEnery},
            publisher = {Lincom-Europa},
            pages = {135--142},
            year = {2002},
        }
        
        @techreport{SimovOseSlav2004,
            author = {Kiril Simov and Petya Osenova and Milena Slavcheva},
            title = {BTB-TR03: BulTreeBank Morphosyntactic Tagset. BulTreeBank Project Technical Report ą 03},
            year = {2004},
            url = {http://www.bultreebank.org/TechRep/BTB-TR03.pdf}
        }
        
        @article{SimOsSimKo2005,
            author = {Kiril Simov and Petya Osenova and Alexander Simov and Milen Kouylekov},
            title = {Design and Implementation of the Bulgarian HPSG-based Treebank},
            journal = {Journal of Research on Language and Computation. Special Issue},
            year = {2005},
            pages = {495--522},
            publisher = {Kluwer Academic Publisher},
        }
        """
        ),
        url="https://universaldependencies.org/",
        data_url="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz",
    ),
    "wikiannbg": DatasetMeta(
        description=textwrap.dedent(
            """\
          The WikiANN dataset (Pan et al. 2017) is a dataset with NER annotations for PER, ORG and LOC. It has been
    constructed using the linked entities in Wikipedia pages for 282 different languages including Danish. The dataset
    can be loaded with the DaNLP package:
    """
        ),
        citation=textwrap.dedent(
            """\
                    @article{pan-x,
            title={Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond},
            author={Xiaoman, Pan and Boliang, Zhang and Jonathan, May and Joel, Nothman and Kevin, Knight and Heng, Ji},
            volume={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers}
            year={2017}
          }"""
        ),
        url="https://github.com/afshinrahimi/mmner",
        data_url=_BASE_DATASET_URL,
    ),
    "ct21t1": DatasetMeta(
        description=textwrap.dedent(
            """\
    The Clef2021 dataset is a dataset in Bulgarian language that comes from 
    Subtask 1A: Check-worthiness of tweets. The task is: 
    Given a tweet, predict whether it is worth fact-checking. We will have not only check-worthiness labels, 
    but also four auxiliary labels. i.e. used to facilitate annotation in 2020, which would enable multi-task learning. 
    We will only evaluate with respect to check-worthiness. 
    This is a classification task. This subtasks runs in 5 languages:
    """
        ),
        citation=textwrap.dedent(
            """\
        @InProceedings{CheckThat:ECIR2021,
          author = {Preslav Nakov and
          Da San Martino, Giovanni and
          Tamer Elsayed and
          Alberto Barr{\'{o}}n{-}Cede{\~{n}}o and
          Rub\'{e}n M\'{i}guez and
          Shaden Shaar and
          Firoj Alam and
          Fatima Haouari and
          Maram Hasanain and
          Nikolay Babulkov and
          Alex Nikolov and
          Shahi, Gautam Kishore and
          Struß, Julia Maria and
          Thomas Mandl},
          title = {The {CLEF}-2021 {CheckThat}! Lab on Detecting Check-Worthy Claims, Previously Fact-Checked Claims, and Fake News},
          booktitle = {Proceedings of the 43rd European Conference on Information Retrieval},
          series = {ECIR~'21},
          pages = {639--649},
          address = {Lucca, Italy},
          month = {March},
          year = {2021},
          url = {https://link.springer.com/chapter/10.1007/978-3-030-72240-1_75},
        }
        """
        ),
        url="https://sites.google.com/view/clef2021-checkthat/tasks/task-1-check-worthiness-estimation",
        data_url=_CLEF_BASE_URL,
    ),
    "crediblenews": DatasetMeta(
        description=textwrap.dedent(
            """\
        Credible news detection dataset.
    """
        ),
        citation=textwrap.dedent(
            """\
        @InProceedings{hardalov2016search,
          author="Hardalov, Momchil
            and Koychev, Ivan
            and Nakov, Preslav",
          editor="Dichev, Christo
            and Agre, Gennady",
          title="In Search of Credible News",
          booktitle="Proceedings of the 17th International Conference 
          on Artificial Intelligence: Methodology, Systems, and Applications",
          year="2016",
          publisher="Springer International Publishing",
          address="Cham",
          pages="172--180",
          isbn="978-3-319-44748-3",
          address="Varna, Bulgaria",
          series="{AIMSA}~'16"
        }
        """
        ),
        url="https://github.com/mhardalov/news-credibility",
        data_url="",
    ),
    "cinexio": DatasetMeta(
        description=textwrap.dedent(
            """\
        Cinexio movie review dataset.
    """
        ),
        citation=textwrap.dedent(
            """\
        @inproceedings{kapukaranov-nakov-2015-fine,
            title = "Fine-Grained Sentiment Analysis for Movie Reviews in {B}ulgarian",
            author = "Kapukaranov, Borislav  and
              Nakov, Preslav",
            booktitle = "Proceedings of the International Conference Recent Advances in Natural Language Processing",
            month = sep,
            year = "2015",
            address = "Hissar, Bulgaria",
            publisher = "INCOMA Ltd. Shoumen, BULGARIA",
            url = "https://aclanthology.org/R15-1036",
            pages = "266--274",
        }
        """
        ),
        url="http://bkapukaranov.github.io/",
        data_url=_BASE_DATASET_URL,
    ),
    "examsbg": DatasetMeta(
        description=textwrap.dedent(
            """\
        EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
        It consists of more than 24,000 high-quality high school exam questions in 16 languages,
        covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
    """
        ),
        citation=textwrap.dedent(
            """\
        @inproceedings{hardalov-etal-2020-exams,
            title = "{EXAMS}: A Multi-subject High School Examinations Dataset for Cross-lingual and Multilingual Question Answering",
            author = "Hardalov, Momchil  and
              Mihaylov, Todor  and
              Zlatkova, Dimitrina  and
              Dinkov, Yoan  and
              Koychev, Ivan  and
              Nakov, Preslav",
            booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
            month = nov,
            year = "2020",
            address = "Online",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2020.emnlp-main.438",
            doi = "10.18653/v1/2020.emnlp-main.438",
            pages = "5427--5444",
        }
        
        @inproceedings{hardalov-etal-2019-beyond,
            title = "Beyond {E}nglish-Only Reading Comprehension: Experiments in Zero-shot Multilingual Transfer for {B}ulgarian",
            author = "Hardalov, Momchil  and
              Koychev, Ivan  and
              Nakov, Preslav",
            booktitle = "Proceedings of the International Conference on Recent Advances in Natural Language Processing",
            month = sep,
            year = "2019",
            address = "Varna, Bulgaria",
            publisher = "INCOMA Ltd.",
            url = "https://aclanthology.org/R19-1053",
            doi = "10.26615/978-954-452-056-4_053",
            pages = "447--459",
            series = "RANLP~'19"
        }
        """
        ),
        url="https://github.com/mhardalov/exams-qa",
        data_url=_BASE_DATASET_URL,
    ),
    "bsnlp": DatasetMeta(
        description=textwrap.dedent(
            """\
            BSNLP is a Workshop that addresses Natural Language Processing (NLP) for 
            the Balto-Slavic languages. The goal of this Workshop is to bring together 
            researchers from academia and industry working on NLP for Balto-Slavic languages. 
            In particular, the Workshop will serve to stimulate research and foster the creation 
            of tools and resources for these languages.
            """
        ),
        citation=textwrap.dedent(
            """\
            @inproceedings{piskorski-etal-2017-first,
                title = "The First Cross-Lingual Challenge on Recognition, Normalization, and Matching of Named Entities in {S}lavic Languages",
                author = "Piskorski, Jakub  and
                  Pivovarova, Lidia  and
                  {\v{S}}najder, Jan  and
                  Steinberger, Josef  and
                  Yangarber, Roman",
                booktitle = "Proceedings of the 6th Workshop on {B}alto-{S}lavic Natural Language Processing",
                month = apr,
                year = "2017",
                address = "Valencia, Spain",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/W17-1412",
                doi = "10.18653/v1/W17-1412",
                pages = "76--85",
            }
            
            @inproceedings{piskorski-etal-2019-second,
                title = "The Second Cross-Lingual Challenge on Recognition, Normalization, Classification, and Linking of Named Entities across {S}lavic Languages",
                author = "Piskorski, Jakub  and
                  Laskova, Laska  and
                  Marci{\'n}czuk, Micha{\l}  and
                  Pivovarova, Lidia  and
                  P{\v{r}}ib{\'a}{\v{n}}, Pavel  and
                  Steinberger, Josef  and
                  Yangarber, Roman",
                booktitle = "Proceedings of the 7th Workshop on Balto-Slavic Natural Language Processing",
                month = aug,
                year = "2019",
                address = "Florence, Italy",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/W19-3709",
                doi = "10.18653/v1/W19-3709",
                pages = "63--74",
            }

            @inproceedings{piskorski-etal-2021-slav,
                title = "Slav-{NER}: the 3rd Cross-lingual Challenge on Recognition, Normalization, Classification, and Linking of Named Entities across {S}lavic Languages",
                author = "Piskorski, Jakub  and
                  Babych, Bogdan  and
                  Kancheva, Zara  and
                  Kanishcheva, Olga  and
                  Lebedeva, Maria  and
                  Marci{\'n}czuk, Micha{\l}  and
                  Nakov, Preslav  and
                  Osenova, Petya  and
                  Pivovarova, Lidia  and
                  Pollak, Senja  and
                  P{\v{r}}ib{\'a}{\v{n}}, Pavel  and
                  Radev, Ivaylo  and
                  Robnik-Sikonja, Marko  and
                  Starko, Vasyl  and
                  Steinberger, Josef  and
                  Yangarber, Roman",
                booktitle = "Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing",
                month = apr,
                year = "2021",
                address = "Kiyv, Ukraine",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2021.bsnlp-1.15",
                pages = "122--133",
            }
            """
        ),
        url="http://bsnlp.cs.helsinki.fi/",
        data_url=_BASE_DATASET_URL,
    ),
    "fakenews": DatasetMeta(
        description=textwrap.dedent(
            """\
            Datathon on fake news detection dataset
            """
        ),
        citation=textwrap.dedent(
            """\
        @inproceedings{karadzhov-etal-2017-built,
            title = "We Built a Fake News / Click Bait Filter: What Happened Next Will Blow Your Mind!",
            author = "Karadzhov, Georgi  and
              Gencheva, Pepa  and
              Nakov, Preslav  and
              Koychev, Ivan",
            booktitle = "Proceedings of the International Conference Recent Advances in Natural Language Processing, {RANLP} 2017",
            month = sep,
            year = "2017",
            address = "Varna, Bulgaria",
            publisher = "INCOMA Ltd.",
            url = "https://doi.org/10.26615/978-954-452-049-6_045",
            doi = "10.26615/978-954-452-049-6_045",
            pages = "334--343",
            series = "RANLP~'17"
        }
        """
        ),
        url="https://gitlab.com/datasciencesociety/case_fake_news",
        data_url=_BASE_DATASET_URL,
    ),
}


def resolve_uri(data_uri: Union[str, Path], target_file: str):
    if isinstance(data_uri, Path):
        return data_uri / target_file

    return f"{data_uri}/{target_file}"


class BgGLUEBaseParser:
    @staticmethod
    @abc.abstractmethod
    def archive_name() -> str:
        raise NotImplementedError("Archive name should be defined")

    @classmethod
    def split_generators(cls, dl_manager=None, config=None):
        data_path = Path(config.data_dir) if config.data_dir else config.data_url
        data_dir = dl_manager.download(resolve_uri(data_path, cls.archive_name()))
        split_filenames = {
            datasets.Split.TRAIN: "train.jsonl",
            datasets.Split.VALIDATION: "dev.jsonl",
            datasets.Split.TEST: "test.jsonl",
        }
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "filepath": dl_manager.iter_archive(data_dir),
                    "filename": split_filenames[split],
                },
            )
            for split in split_filenames
        ]


class BsnlpParser(BgGLUEBaseParser):

    features = {
        "ID": datasets.Value("string"),
        "tokens": datasets.Sequence(datasets.Value("string")),
        "ner_tags": datasets.Sequence(
            datasets.features.ClassLabel(
                names=[
                    "O",
                    "B-PRO",
                    "I-PRO",
                    "B-LOC",
                    "I-LOC",
                    "B-ORG",
                    "I-ORG",
                    "B-PER",
                    "I-PER",
                    "B-EVT",
                    "I-EVT",
                ]
            )
        ),
        "langs": datasets.Sequence(datasets.Value("string")),
    }

    @staticmethod
    def archive_name() -> str:
        return "bsnlp.tar.gz"

    @staticmethod
    def generate_examples(config=None, filepath=None, filename=None):
        idx = 0

        for path, file in filepath:
            if path.endswith(filename):
                lines = (line.decode("utf-8") for line in file)
                for line in lines:
                    idx += 1
                    yield idx, json.loads(line)


class FakeNewsParser(BgGLUEBaseParser):

    features = {
        "title": datasets.Value("string"),
        "content": datasets.Value("string"),
        "date_published": datasets.Value("string"),
        "url": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=["credible", "fake"]),
    }

    @staticmethod
    def archive_name() -> str:
        return "fakenews.tar.gz"

    @staticmethod
    def generate_examples(config=None, filepath=None, filename=None):
        idx = 0

        for path, file in filepath:
            if path.endswith(filename):
                lines = (line.decode("utf-8") for line in file)

                for example in lines:
                    idx += 1
                    example = json.loads(example)
                    example["label"] = int(example["fake_news"])
                    del example["fake_news"]

                    yield idx, example


class CredibleNewsParser(BgGLUEBaseParser):

    features = {
        "key": datasets.Value("string"),
        "title": datasets.Value("string"),
        "content": datasets.Value("string"),
        "publishDate": datasets.Value("string"),
        "source": datasets.Value("string"),
        "category": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=["credible", "humorous"]),
    }

    @staticmethod
    def archive_name() -> str:
        return "credible_news.tar.gz"

    @staticmethod
    def generate_examples(config=None, filepath=None, filename=None):
        idx = 0

        for path, file in filepath:
            if path.endswith(filename):
                lines = (line.decode("utf-8") for line in file)
                for line in lines:
                    idx += 1
                    example = json.loads(line)

                    yield idx, example


class CheckThat21T1Parser:
    """Clef2021: Subtask 1A: Check-worthiness of tweets."""

    features = {
        "tweet_id": datasets.Value("string"),
        "id_str": datasets.Value("string"),
        "topic_id": datasets.Value("string"),
        "tweet_text": datasets.Value("string"),
        "labels": datasets.ClassLabel(names=["normal", "check-worthy"]),
    }

    @staticmethod
    def split_generators(dl_manager=None, config=None):
        if config.data_dir:
            data_dir = Path(config.data_dir)
            split_urls = {
                "train_dev": data_dir / "v1.zip",
                "test": data_dir / "subtask-1a--bulgarian.zip",
            }
        else:
            split_urls = {
                "train_dev": f"{_CLEF_BASE_URL}/data/subtask-1a--bulgarian/v1.zip",
                "test": f"{_CLEF_BASE_URL}/test-input/subtask-1a--bulgarian.zip",
            }

        data_dir = dl_manager.download_and_extract(split_urls)
        split_filenames = {
            datasets.Split.TRAIN: "v1/dataset_train_v1_bulgarian.tsv",
            datasets.Split.VALIDATION: "v1/dataset_dev_v1_bulgarian.tsv",
            datasets.Split.TEST: "subtask-1a--bulgarian/dataset_test_input_bulgarian.tsv",
        }
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "filepath": data_dir[
                        "train_dev" if not split == datasets.Split.TEST else "test"
                    ],
                    "filename": split_filenames[split],
                },
            )
            for split in split_filenames
        ]

    @staticmethod
    def generate_examples(config=None, filepath=None, filename=None):
        with Path(filepath).joinpath(filename).open(encoding="utf-8") as fp:
            data = csv.DictReader(fp, delimiter="\t", quoting=csv.QUOTE_NONE)
            for id_, row in enumerate(data):
                row["id_str"] = str(row["tweet_id"])
                # In the validation set tsv file the field is called `claim_worthiness`
                label_field = (
                    "check_worthiness"
                    if "check_worthiness" in row
                    else "claim_worthiness"
                )

                # The test set labels are not shared as part of the benchmark, and we set them to 0.
                row["labels"] = int(row.get(label_field, 0))

                # Remove unused fields from the dictionary
                for field in (
                    "check_worthiness",
                    "claim_worthiness",
                    "claim",
                    "tweet_url",
                ):
                    if field in row:
                        del row[field]

                yield id_, row


class CinexioParser(BgGLUEBaseParser):
    features = {
        "ID": datasets.Value("string"),
        "Cinexio_URL": datasets.Value("string"),
        "Comment": datasets.Value("string"),
        "Date": datasets.Value("string"),
        "User_Rating": datasets.Value("float32"),
        "Category": datasets.Value("int32"),
        "label": datasets.Value("float32"),
    }

    @staticmethod
    def archive_name() -> str:
        return "cinexio.tar.gz"

    @staticmethod
    def generate_examples(config=None, filepath=None, filename=None):
        idx = 0
        for path, file in filepath:
            if path.endswith(filename):
                lines = (line.decode("utf-8") for line in file)
                for line in lines:
                    idx += 1
                    example = json.loads(line)
                    example["label"] = example["User_Rating"]

                    yield idx, example


class ExamsParser(BgGLUEBaseParser):
    """Exams dataset"""

    features = {
        "id": datasets.Value("string"),
        "question": {
            "stem": datasets.Value("string"),
            "choices": datasets.Sequence(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    "para": datasets.Value("string"),
                }
            ),
        },
        "answerKey": datasets.Value("string"),
        "info": {
            "grade": datasets.Value("int32"),
            "subject": datasets.Value("string"),
            "language": datasets.Value("string"),
        },
    }

    @staticmethod
    def archive_name() -> str:
        return "exams.tar.gz"

    @staticmethod
    def generate_examples(config=None, filepath=None, filename=None):
        for path, file in filepath:
            if path.endswith(filename):
                for id_, line in enumerate(file):
                    line_dict = json.loads(line.strip())

                    if "choices" in line_dict:
                        del line_dict["choices"]

                    for choice in line_dict["question"]["choices"]:
                        choice["para"] = choice.get("para", "")
                    yield id_, line_dict
                break


class WikiannBgParser:
    """WikiANN is a multilingual named entity recognition dataset consisting of Wikipedia articles annotated with LOC, PER, and ORG tags"""

    features = {
        "tokens": datasets.Sequence(datasets.Value("string")),
        "ner_tags": datasets.Sequence(
            datasets.features.ClassLabel(
                names=[
                    "O",
                    "B-PER",
                    "I-PER",
                    "B-ORG",
                    "I-ORG",
                    "B-LOC",
                    "I-LOC",
                ]
            )
        ),
        "langs": datasets.Sequence(datasets.Value("string")),
        "spans": datasets.Sequence(datasets.Value("string")),
    }

    @staticmethod
    def _tags_to_spans(tags):
        """Convert tags to spans."""
        spans = set()
        span_start = 0
        span_end = 0
        active_conll_tag = None
        for index, string_tag in enumerate(tags):
            # Actual BIO tag.
            bio_tag = string_tag[0]
            assert bio_tag in ["B", "I", "O"], "Invalid Tag"
            conll_tag = string_tag[2:]
            if bio_tag == "O":
                # The span has ended.
                if active_conll_tag:
                    spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = None
                # We don't care about tags we are
                # told to ignore, so we do nothing.
                continue
            elif bio_tag == "B":
                # We are entering a new span; reset indices and active tag to new span.
                if active_conll_tag:
                    spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
            elif bio_tag == "I" and conll_tag == active_conll_tag:
                # We're inside a span.
                span_end += 1
            else:
                # This is the case the bio label is an "I", but either:
                # 1) the span hasn't started - i.e. an ill formed span.
                # 2) We have IOB1 tagging scheme.
                # We'll process the previous span if it exists, but also include this
                # span. This is important, because otherwise, a model may get a perfect
                # F1 score whilst still including false positive ill-formed spans.
                if active_conll_tag:
                    spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
        # Last token might have been a part of a valid span.
        if active_conll_tag:
            spans.add((active_conll_tag, (span_start, span_end)))
        # Return sorted list of spans
        return sorted(list(spans), key=lambda x: x[1][0])

    @staticmethod
    def _get_spans(tokens, tags):
        """Convert tags to textspans."""
        spans = WikiannBgParser._tags_to_spans(tags)
        text_spans = [
            x[0] + ": " + " ".join([tokens[i] for i in range(x[1][0], x[1][1] + 1)])
            for x in spans
        ]
        if not text_spans:
            text_spans = ["None"]
        return text_spans

    @staticmethod
    def split_generators(dl_manager=None, config=None):
        data_path = Path(config.data_dir) if config.data_dir else config.data_url
        data_dir = dl_manager.download(resolve_uri(data_path, "wikiann_bg.tar.gz"))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filename": "dev",
                    "filepath": dl_manager.iter_archive(data_dir),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filename": "test",
                    "filepath": dl_manager.iter_archive(data_dir),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filename": "train",
                    "filepath": dl_manager.iter_archive(data_dir),
                },
            ),
        ]

    @staticmethod
    def generate_examples(config=None, filepath=None, filename=None):
        """Reads line by line format of the NER dataset and generates examples.
        Input Format:
        en:rick  B-PER
        en:and  O
        en:morty  B-PER
        en:are  O
        en:cool  O
        en:.  O
        Output Format:
        {
        'tokens': ["rick", "and", "morty", "are", "cool", "."],
        'ner_tags': ["B-PER", "O" , "B-PER", "O", "O", "O"],
        'langs': ["en", "en", "en", "en", "en", "en"]
        'spans': ["PER: rick", "PER: morty"]
        }
        Args:
            filepath: Path to file with line by line NER format.
        Returns:
            Examples with the format listed above.
        """
        guid_index = 1
        for path, f in filepath:
            if path == filename:
                tokens = []
                ner_tags = []
                langs = []
                for line in f:
                    line = line.decode("utf-8")
                    if line == "" or line == "\n":
                        if tokens:
                            spans = WikiannBgParser._get_spans(tokens, ner_tags)
                            yield guid_index, {
                                "tokens": tokens,
                                "ner_tags": ner_tags,
                                "langs": langs,
                                "spans": spans,
                            }
                            guid_index += 1
                            tokens = []
                            ner_tags = []
                            langs = []
                    else:
                        # wikiann data is tab separated
                        splits = line.split("\t")
                        # strip out en: prefix
                        langs.append(splits[0].split(":")[0])
                        tokens.append(":".join(splits[0].split(":")[1:]))
                        if len(splits) > 1:
                            ner_tags.append(splits[-1].replace("\n", ""))
                        else:
                            # examples have no label in test set
                            ner_tags.append("O")
                break


class UdposParser:

    features = {
        "tokens": datasets.Sequence(datasets.Value("string")),
        "pos_tags": datasets.Sequence(
            datasets.features.ClassLabel(
                names=[
                    "ADJ",
                    "ADP",
                    "ADV",
                    "AUX",
                    "CCONJ",
                    "DET",
                    "INTJ",
                    "NOUN",
                    "NUM",
                    "PART",
                    "PRON",
                    "PROPN",
                    "PUNCT",
                    "SCONJ",
                    "SYM",
                    "VERB",
                    "X",
                ]
            )
        ),
    }

    @staticmethod
    def split_generators(dl_manager=None, config=None):
        data_path = Path(config.data_dir) if config.data_dir else config.data_url
        archive = dl_manager.download(data_path)
        split_names = {
            datasets.Split.TRAIN: "train",
            datasets.Split.VALIDATION: "dev",
            datasets.Split.TEST: "test",
        }
        split_generators = {
            split: datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "filepath": dl_manager.iter_archive(archive),
                    "split": split_names[split],
                },
            )
            for split in split_names
        }

        return [
            split_generators["train"],
            split_generators["validation"],
            split_generators["test"],
        ]

    @staticmethod
    def generate_examples(config=None, filepath=None, split=None):
        lang = "Bulgarian"
        idx = 0
        for path, file in filepath:
            if f"_{lang}" in path and split in path and path.endswith(".conllu"):
                # For lang other than [see below], we exclude Arabic-NYUAD which does not contains any words, only _
                if (
                    lang in ["Kazakh", "Tagalog", "Thai", "Yoruba"]
                    or "NYUAD" not in path
                ):
                    lines = (line.decode("utf-8") for line in file)
                    data = csv.reader(lines, delimiter="\t", quoting=csv.QUOTE_NONE)
                    tokens = []
                    pos_tags = []
                    for id_row, row in enumerate(data):
                        if len(row) >= 10 and row[1] != "_" and row[3] != "_":
                            tokens.append(row[1])
                            pos_tags.append(row[3])
                        if len(row) == 0 and len(tokens) > 0:
                            yield idx, {
                                "tokens": tokens,
                                "pos_tags": pos_tags,
                            }
                            idx += 1
                            tokens = []
                            pos_tags = []


class XnliParser:
    """XNLI: The Cross-Lingual NLI Corpus. Version 1.0."""

    features = {
        "premise": datasets.Value("string"),
        "hypothesis": datasets.Value("string"),
        "label": datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
    }

    @staticmethod
    def split_generators(dl_manager=None, config=None):
        data_path = Path(config.data_dir) if config.data_dir else config.data_url
        dl_dirs = dl_manager.download_and_extract(
            {
                "train_data": resolve_uri(data_path, "XNLI-MT-1.0.zip"),
                "testval_data": resolve_uri(data_path, "XNLI-1.0.zip"),
            }
        )
        train_dir = Path(dl_dirs["train_data"]) / "XNLI-MT-1.0" / "multinli"
        testval_dir = Path(dl_dirs["testval_data"]) / "XNLI-1.0"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": [train_dir / f"multinli.train.bg.tsv"],
                    "data_format": "XNLI-MT",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": [testval_dir / "xnli.test.tsv"],
                    "data_format": "XNLI",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": [testval_dir / "xnli.dev.tsv"],
                    "data_format": "XNLI",
                },
            ),
        ]

    @staticmethod
    def generate_examples(config, filepath, data_format=None):
        """This function returns the examples in the raw (text) form."""
        if data_format == "XNLI-MT":
            for file_idx, file in enumerate(filepath):
                with open(file, encoding="utf-8") as fp:
                    reader = csv.DictReader(fp, delimiter="\t", quoting=csv.QUOTE_NONE)
                    for row_idx, row in enumerate(reader):
                        key = str(file_idx) + "_" + str(row_idx)
                        yield key, {
                            "premise": row["premise"],
                            "hypothesis": row["hypo"],
                            "label": row["label"].replace(
                                "contradictory", "contradiction"
                            ),
                        }
        else:
            for file_idx, file in enumerate(filepath):
                with open(file, encoding="utf-8") as fp:
                    reader = csv.DictReader(fp, delimiter="\t", quoting=csv.QUOTE_NONE)
                    for row in reader:
                        if row["language"] == "bg":
                            yield row["pairID"], {
                                "premise": row["sentence1"],
                                "hypothesis": row["sentence2"],
                                "label": row["gold_label"],
                            }


class BgglueConfig(datasets.BuilderConfig):
    """BuilderConfig for Break"""

    def __init__(self, data_url, citation, url, text_features, **kwargs):
        """
        Args:
            text_features: `dict[string, string]`, map from the name of the feature
        dict for each text field to the name of the column in the tsv file
            label_column:
            label_classes
            **kwargs: keyword arguments forwarded to super.
        """
        super(BgglueConfig, self).__init__(
            version=datasets.Version("1.0.0", ""), **kwargs
        )
        self.text_features = text_features
        self.data_url = data_url
        self.citation = citation
        self.url = url


_PARSERS = {
    "bsnlp": BsnlpParser,
    "fakenews": FakeNewsParser,
    "ct21t1": CheckThat21T1Parser,
    "xnlibg": XnliParser,
    "udep": UdposParser,
    "wikiannbg": WikiannBgParser,
    "crediblenews": CredibleNewsParser,
    "cinexio": CinexioParser,
    "examsbg": ExamsParser,
}


class Bgglue(datasets.GeneratorBasedBuilder):
    """
    bgGLUE (Bulgarian General Language Understanding Evaluation) is a benchmark for evaluating language models
    on Natural Language Understanding (NLU) tasks in Bulgarian. The benchmark includes NLU tasks targeting a variety
    of NLP problems (e.g., natural language inference, fact-checking, named entity recognition, sentiment analysis,
    question answering, etc.) and machine learning tasks (sequence labeling, document-level classification,
    and regression).
    """

    VERSION = datasets.Version("0.1.0")

    BUILDER_CONFIGS = [
        BgglueConfig(
            name=name,
            description=meta.description,
            citation=meta.citation,
            text_features=_PARSERS[name].features,
            data_url=meta.data_url,
            url=meta.url,
        )
        for name, meta in _DATASETS_META.items()
    ]

    def _info(self):
        features = _PARSERS[self.config.name].features

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=f"{self.config.description}\n{_DESCRIPTION}",
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                # These are the features of your dataset like images, labels ...
                features
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=f"{_BASE_URL}\t{self.config.url}",
            citation=f"{self.config.citation}\n{_CITATION}",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        return _PARSERS[self.config.name].split_generators(
            dl_manager=dl_manager, config=self.config
        )

    def _generate_examples(self, filepath=None, **kwargs):
        """Yields examples."""
        return _PARSERS[self.config.name].generate_examples(
            config=self.config, filepath=filepath, **kwargs
        )

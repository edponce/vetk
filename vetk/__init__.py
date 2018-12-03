"""Vector Embedding Toolkit (VETk) package.

Todo:
    Improve handling of module imports vs special attributes if using
    non-standard libraries. Currently applicable when runnign tox environments
    that do not include the install_requirements.txt:
        * Use try-except in __init__.py (bad hack)
        * Use explicit values in setup.py and docs/conf.py
        * Include install_requirements.txt in tox environment (e.g., docs)
"""


from .embeddings import WordEmbedding
from .analytics import WordEmbeddingAnalysis


__all__ = [
    'WordEmbedding',
    'WordEmbeddingAnalysis'
]


__title__ = "Vector Embedding Toolkit (VETk)"
__name__ = "VETk"
__version__ = "0.2.0"
__description__ = "Analytics framework for vector embeddings"
__keywords__ = [
    "vector embedding",
    "word embedding",
    "word2vec",
    "natural language processing"
]
__url__ = "https://github.com/edponce/vetk"
__author__ = "Eduardo Ponce, Oak Ridge National Laboratory, Oak Ridge, TN"
__author_email__ = "edponce2010@gmail.com"
__license__ = "MIT"
__copyright__ = "2018 Eduardo Ponce"

"""Vector Embedding Toolkit (VETk) module"""


from .embeddings import WordEmbedding
from .analytics import WordEmbeddingAnalysis


__all__ = [
    'WordEmbedding',
    'WordEmbeddingAnalysis'
    ]


__title__ = 'Vector Embedding Toolkit (VETk)'
__name__ = 'VETk'
__version__ = '0.5'
__description__ = 'Analytics framework for vector embeddings'
__keywords__ = [
    'vector embedding',
    'word embedding',
    'word2vec',
    'natural language processing'
    ]
__url__ = 'https://code.ornl.gov/VA-OSL/vector-embedding-toolkit'
__author__ = 'Eduardo Ponce, Oak Ridge National Laboratory, Oak Ridge, TN'
__author_email__ = 'poncemojicae@ornl.gov'
__license__ = 'MIT'
__copyright__ = '2018 Eduardo Ponce'

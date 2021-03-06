import unittest
import vetkit as vk


class TestConfigurationFiles(unittest.TestCase):

    def test_emptyWordEmbedding(self):
        ve = vk.WordEmbedding()
        self.assertTrue(len(ve.vocabulary) == 0 and len(ve.vectors) == 0)

    def test_loadWord2Vec(self):
        ve = vk.WordEmbedding('vetkit/data/word2vec/364words_100dim_1threads.vec')
        self.assertTrue(ve.vectors.shape[1] == 364 and
                        ve.vectors.shape[2] == 100 and
                        len(ve.vocabulary) == 364)


if __name__ == '__main__':
    unittest.main()

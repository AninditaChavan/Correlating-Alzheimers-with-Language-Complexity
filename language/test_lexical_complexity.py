import unittest

from nltk.tree import Tree
from lexical_complexity import LexicalComplexity

class LexicalComplexityTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(LexicalComplexityTest, self).__init__(*args, **kwargs)
   
        self.complexity = LexicalComplexity()

    def test_calculate_ttr(self):
        sent = "Colorless green ideas sleep furiously. Trees leaves are green."
        expected = 0.818
        actual = self.complexity.calculate_ttr(sent)
        print("score = ", actual)
        self.assertEqual(expected, round(actual,3))


    def test_honore_statistics(self):
        sent = "Colorless green ideas sleep furiously. Trees leaves are green."

        expected = 390.197
        actual = self.complexity.calculate_honore_statistics(sent)
        self.assertEqual(expected, round(actual, 3))

    def test_automatic_readability_index(self):
        sent = "Colorless green ideas sleep furiously. Trees leaves are green."

        expected =  8.033
        actual = self.complexity.automatic_readability_index(sent)
        print("score = ", actual)
        self.assertEqual(expected, round(actual, 3))

    def test_brunet_index(self):
        sent = "Colorless green ideas sleep furiously. Trees leaves are green."

        expected =  8.265
        actual = self.complexity.calculate_brunet_index(sent)
        self.assertEqual(expected, round(actual,3))

    def test_coleman_liau_index(self):
        sent = "Colorless green ideas sleep furiously.  Trees leaves are green."

        expected = 11.596
        actual = self.complexity.calculate_coleman_liau_index(sent)
        self.assertEqual(expected, round(actual, 3))


if __name__ == '__main__':
    unittest.main()
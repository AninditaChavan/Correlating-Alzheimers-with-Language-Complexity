import unittest

from nltk.tree import Tree
from syntactic_complexity import Complexity

class ComplexityTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ComplexityTestCase, self).__init__(*args, **kwargs)
   
        self.complexity = Complexity()

    def test_get_mean_yngve(self):
        sent = "Colorless green ideas sleep furiously"

        parse = ['( (S (NP (NNP Colorless) (JJ green) (NNS ideas)) (VP (VBP sleep) (ADVP (RB furiously)))) )']
        expected = 1.4
        actual = self.complexity.get_mean_yngve(parse)
        print("score = ", actual)
        self.assertEqual(expected, actual)


    def test_yngve_redux(self):
        sent = "Colorless green ideas sleep furiously"
        parse = ['( (S (NP (NNP Colorless) (JJ green) (NNS ideas)) (VP (VBP sleep) (ADVP (RB furiously)))) )']

        expected = [7.0, 5.0]
        actual = self.complexity.yngve_redux(parse[0])
        self.assertEqual(expected, actual)

    def test_calc_yngve_score(self):
        sent = "Colorless green ideas sleep furiously"
        #parse = ['( (S (NP (NNP Colorless) (JJ green) (NNS ideas)) (VP (VBP sleep) (ADVP (RB furiously)))) )']

        parse = ['(S (RB So) (NP (NP (CD 4) (JJ o) (NN ’) (NN clock)) (PP (IN in) (NP (DT the) (NN morning)))))']
        expected = 13
        actual = self.complexity.calc_yngve_score(Tree.fromstring(parse[0]), 0)
        print("score = ", actual)
        self.assertEqual(expected, actual)

    def test_get_mean_frazier(self):
        sent = "Colorless green ideas sleep furiously"
        parse = ['( (S (NP (NNP Colorless) (JJ green) (NNS ideas)) (VP (VBP sleep) (ADVP (RB furiously)))) )']

        expected = 0.9
        actual = self.complexity.get_mean_frazier(parse)
        self.assertEqual(expected, actual)


    def test_calc_frazier_score(self):
        sent = "Colorless green ideas sleep furiously"
        parse = ['( (S (NP (NNP Colorless) (JJ green) (NNS ideas)) (VP (VBP sleep) (ADVP (RB furiously)))) )']

        expected = 4.5
        actual = self.complexity.calc_frazier_score(Tree.fromstring(parse[0]), 0, '')
        self.assertEqual(expected, actual)

        expected = -1
        actual = self.complexity.calc_frazier_score("Hi!", 0, '')
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
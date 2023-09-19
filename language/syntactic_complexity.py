from nltk.tree import Tree

import spacy
import numpy as np

en_nlp = spacy.load('en_core_web_sm')

class Complexity():
    def __init__(self):
        self.total_score = 0
        
    
    def extract_sentences(self, trees):
        stack = 0
        startIndex = None
        sentences = []

        for i, c in enumerate(trees):
            if c == '(':
                if stack == 0:
                    startIndex = i + 1 # string to extract starts one index later

                # push to stack
                stack += 1
            elif c == ')':
                # pop stack
                stack -= 1

                if stack == 0:
                    sentences.append(trees[startIndex-1:i+1])

        return sentences

    def word_score(self, tree):
        """ Calculate the word score a tree. """
        if type(tree) == str:
            return 1
        else:
            score = 0
            for child in tree:
                score += self.word_score(child)
            return score


    def yngve_redux(self, treestring):
        """
        For the given treestring, return the word count and the Yngve score.
        """
        tree = Tree.fromstring(treestring)
        total = float(self.calc_yngve_score(tree, 0))
        words = float(self.word_score(tree))

        return [total, words]

    def calc_yngve_score(self, tree, parent=0):
        """ Calculate the Yngve score for a given tree. """
        if type(tree) == str:
            return parent
        else:
            c = 0
            for i, child in enumerate(reversed(tree)):
                c += self.calc_yngve_score(child, parent + i)
            return c

    
    def get_mean_yngve(self, treestrings):
        """ Average all of the yngve scores for the given input. """
        c = 0
        total = 0
        if type(treestrings) != list:
            raise ValueError('Input to get_mean_yngve() must be a list of strings.')

        for treestring in treestrings:
            results = self.yngve_redux(treestring)
            total += results[0]
            c += results[1]

        try:
            score = float(total / c)
        except ZeroDivisionError:
            print('ZeroDivisionError for Yngve calculation.')
            score = 0.0

        return score


    def is_sent(self, value):
        """ Determine if the given string is a sentence. """
        return len(value) > 0 and value[0] == "S"

    def calc_frazier_score(self, tree, parent, parent_label):
        """ Calculate the Frazier scores for the given input. """
        current_label = ''
        if type(tree) == str:
            return parent-1
        else:
            c = 0
            for i, child in enumerate(tree):
                score = 0
                if i == 0:
                    current_label = tree.label()
                    if self.is_sent(current_label):
                        score = (0 if self.is_sent(parent_label) else parent + 1.5)
                    elif current_label != '' and current_label != "ROOT" and current_label != "TOP":
                        score = parent+1
                c += self.calc_frazier_score(child, score, current_label)
            return c
            
    def get_mean_frazier(self, treestrings):
        """ Average all of the Frazier scores for the given input. """
        sentences, total_frazier_score, total_word_count = 0, 0, 0
        if type(treestrings) != list:
            raise ValueError('Input to get_mean_frazier() must be a list of '
                            'strings.')
        for tree_line in treestrings:
            if tree_line.strip() == "":
                continue
            tree = Tree.fromstring(tree_line)
            sentences += 1
            raw_frazier_score = self.calc_frazier_score(tree, 0, "")

            total_word_count += self.word_score(tree)
            total_frazier_score += raw_frazier_score

        try:
            score = float(total_frazier_score) / float(total_word_count)
        except ZeroDivisionError:
            print('ZeroDivisionError for Frazier calculation.')
            score = 0.0

        return score

    def dependency_length(self, node, depth):
        if node.n_lefts + node.n_rights > 0:
            return max(self.dependency_length(child, depth + 1) for child in node.children)
        else:
            return depth

    def syntactic_complexity(self, trees):
        sentences = self.extract_sentences(trees)

        yngve_scores = []
        frazier_scores = []
        sdl = []
        for s in sentences:
            yngve_scores.append(self.get_mean_yngve(sentences))
            frazier_scores.append(self.get_mean_frazier(sentences))
            doc = en_nlp(s)
            for sent in doc.sents:
                sdl.append(self.dependency_length(sent.root, 0))

        yngve_mean = np.mean(yngve_scores)
        frazier_mean = np.mean(frazier_scores)
        sdl_mean = np.mean(sdl)

        print("Mean Yngve score = ", yngve_mean)
        print("Mean Frazier score = ", frazier_mean)
        print("Mean SDL score = ", sdl_mean)

        return yngve_mean, frazier_mean, sdl_mean
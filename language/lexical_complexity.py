import nltk
import numpy as np
import pandas as pd
import math
from nltk.corpus import brown
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer 
lmtzr = WordNetLemmatizer()


'''Calculating type-token ratio - documents lexical richness or variety in vocabulary. 
total number of unique words divided by total number of tokens. Word lemmatization reduces 
each word to its root. eg. studying and studies is considered as the same word (vocabulary). Hence it is more
accurate'''

class LexicalComplexity():

    def __init__(self):
        self.noun_list = ['NN', 'NNS', 'NNP', 'NNPS']
        self.verb_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP']
        self.grammar = r"""
            DTR: {<DT><DT>}
            NP: {<DT>?<JJ>*<NN.*>} 
            PP: {<IN><NP>} 
            VPG: {<VBG><NP | PP>}
            VP: {<V.*><NP | PP>}     
            CLAUSE: {<NP><VP>} 
            """  

    def get_freq_token_type(self, sentence):
        text = nltk.word_tokenize(sentence)
        num_tokens = len(text)
        freq_token_type = Counter(text)
        return freq_token_type

    def get_freq_token_type_lem(self, sentence):
        text = nltk.word_tokenize(sentence)
        numtokens = len(text)
        tag_info = np.array(nltk.pos_tag(text))
        text_root = [lmtzr.lemmatize(j) for indexj, j in enumerate(text)]
        for indexj, j in enumerate(text):
            if tag_info[indexj,1] in self.noun_list:
                text_root[indexj] = lmtzr.lemmatize(j) 
            elif tag_info[indexj,1] in self.verb_list:
                text_root[indexj] = lmtzr.lemmatize(j,'v')
        freq_lemmtoken_type = Counter(text_root)
        return freq_lemmtoken_type

    def calculate_ttr(self, sentence):
        text = nltk.word_tokenize(sentence)
        num_tokens = len(text)
        freq_token_type = self.get_freq_token_type(sentence)
        num_types = len(freq_token_type)
        ttr = float(num_types)/num_tokens
        print("Type to token ratio:",ttr)
        return ttr

    #Lemmatizes each word eg. studying and studies is the same thing
    def calculate_ttr_lematized(self, sentence):
        text = nltk.word_tokenize(sentence)
        numtokens = len(text)
        freq_lemmtoken_type = self.get_freq_token_type_lem(sentence)
        num_types_lem = len(freq_lemmtoken_type)
        ttr_lemmatized = float(num_types_lem)/numtokens  
        print("Type to token ratio lemmatized:",ttr_lemmatized)
        return ttr_lemmatized

    ''' Honore's statistic is based on the notion that larger the number of words used
    by speaker that occur only once, richer his overall lexicon is. Words spoken only once and total vocabulary used 
    are linearly associated. R=100×log(N/(1−V 1/V))where N is the total text length. Higher values correspond to
    a richer vocabulary. As with standardized word entropy, stemming is done on words and only the stems are considered. '''
    def calculate_honore_statistics(self, sentence):
        text = nltk.word_tokenize(sentence)
        numtokens = len(text)
        freq_token_root = self.get_freq_token_type_lem(sentence)
        freq_token_type = Counter(text)  
        v = len(freq_token_type)
        occur_once = 0
        for j in freq_token_root:
            if freq_token_root[j] == 1:
                occur_once = occur_once + 1
        v1 = occur_once
        
        honoroe_stats = 100 * math.log(numtokens / (1 - (v1/v)))
        print("Honoroe's statistics:",honoroe_stats)
        return honoroe_stats

    def automatic_readability_index(self, sentence):
        num_char = len([c for c in sentence if c.isdigit() or c.isalpha()])
        num_words = len([word for word in sentence.split(' ') if not word=='' and not word=='.'])
        num_sentences = sentence.count('.') + sentence.count('?')
        ARI = 4.71*(num_char/num_words) + 0.5*(num_words/num_sentences) - 21.43
        print("Automatic Readability Index:",ARI)
        return ARI
        
    def calculate_brunet_index(self, sentence):
        text = nltk.word_tokenize(sentence)
        numtokens = len(text)
        freq_lemmtoken_type = self.get_freq_token_type_lem(sentence)
        vl = len(freq_lemmtoken_type)
        brunet_index = float(vl)**(numtokens**-0.0165)
        print("Brunet's index:",brunet_index)
        return brunet_index
        
    def calculate_coleman_liau_index(self, sentence):
        num_char = len([c for c in sentence if c.isdigit() or c.isalpha()])
        num_words = len([word for word in sentence.split(' ') if not word=='' and not word=='.'])
        num_sentences = sentence.count('.') + sentence.count('?')
        L = (num_char/num_words)*100
        S = (num_sentences/num_words)*100
        CLI = 0.0588*L - 0.296*S - 15.8 
        print("Coleman Liau's index:",CLI)
        return CLI

    def calculate_word_to_sentence_ration(self, sentence):
        num_words = len([word for word in sentence.split(' ') if not word=='' and not word=='.'])
        num_sentences = sentence.count('.') + sentence.count('?')
        word_sentence_ratio = num_words/num_sentences
        print("Word to sentence ratio:",word_sentence_ratio)
        return word_sentence_ratio
        
    def get_frequency_counts(self, sentence):
        text = nltk.word_tokenize(sentence)
        tag_info = np.array(nltk.pos_tag(text))
        tag_fd = nltk.FreqDist(tag for i, (word, tag) in enumerate(tag_info))
        freq_tag = tag_fd.most_common()
        sent = nltk.pos_tag(text)
        cp = nltk.RegexpParser(self.grammar)
        phrase_type = cp.parse(sent)  
        # print(phrase_type)
        prp_count = sum([pos[1] for pos in freq_tag if pos[0]=='PRP' or pos[0]=='PRP$'])
        noun_count = sum([pos[1] for pos in freq_tag if pos[0] in noun_list])
        vg_count = sum([pos[1] for pos in freq_tag if pos[0]=='VBG'])

        # ------- Pronoun-to-Noun ratio -------
        if noun_count != 0:
            prp_noun_ratio = prp_count/noun_count
        else:
            prp_noun_ratio = prp_count

        # Noun phrase, Verb phrase, Verb gerund phrase frequency        
        NP_count = 0
        VP_count = 0
        VGP_count = 0
        for index_t, t in enumerate(phrase_type):
            if not isinstance(phrase_type[index_t],tuple):
                if phrase_type[index_t].label() == 'NP':
                    NP_count = NP_count + 1
                elif phrase_type[index_t].label() == 'VP': 
                    VP_count = VP_count + 1
                elif phrase_type[index_t].label() == 'VGP':
                    VGP_count = VGP_count + 1

    def get_lexical_measures(self, content):
        ttr_lematized = self.calculate_ttr_lematized(content)
        ttr = self.calculate_ttr(content)
        honore_statistics = self.calculate_honore_statistics(content)
        ARI = self.automatic_readability_index(content)
        brunet_index = self.calculate_brunet_index(content)
        CLI = self.calculate_coleman_liau_index(content)

        return ttr_lematized, ttr, honore_statistics, ARI, brunet_index, CLI
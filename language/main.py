import argparse

# Punctuation and spell correct dependencies
from nemo.collections.nlp.models import PunctuationCapitalizationModel
import contextualSpellCheck
import spacy
from textblob import TextBlob

import benepar, spacy
from nltk.tree import Tree
import numpy as np
from syntactic_complexity import Complexity
from lexical_complexity import LexicalComplexity


def prepare_input(path, spell_check = False):
    nlp = spacy.load("en_core_web_sm") 

    contextualSpellCheck.add_to_pipe(nlp)

    # Download and load the pre-trained BERT-based model
    model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")

    print("\nReading input file and preparing punctuated and spell-corrected text...")

    # Read raw file provided by the ASR team
    with open(path, "r") as f:
        text = f.read().replace('\n', '')
        
    # Punctuate the text using nemo
    punkted = model.add_punctuation_capitalization([text.lower()])[0]

    # Using contextual spell check to correct spellings
    if spell_check:
        doc = nlp(punkted.lower())
        print("\n\n Spell-correction: FINISHED")

    else:
        doc = nlp(punkted)

    refined_text = doc._.outcome_spellCheck

    f = open('punctuated.txt', "w")
    f.write(str(refined_text))
    f.close()

    print("\n\nPunctuation: FINISHED")
    return refined_text

def constituency_parser(refined_text):
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

    print("\nParsing punctuated text...")

    doc = nlp(refined_text)

    sentences = list(doc.sents)

    parsed = ""
    for s in sentences:
        parsed += s._.parse_string

    with open("parsed.txt", "w") as parsed_output_file:
        parsed_output_file.write(parsed)
    print("\nParsing FINISHED")

    return parsed


def main():

    parser = argparse.ArgumentParser(description='baseline')
    parser.add_argument('--path', type=str, default='./input.txt')
    parser.add_argument('--normalize', action = 'store_true')

    args = parser.parse_args()
    path = args.path
    spell_check = args.normalize

    # Punctuate and spell-correct text
    refined_text = prepare_input(path, spell_check)

    # Calculating syntactic complexity
    parser_output = constituency_parser(refined_text) # generating constituency parse tree

    syntactic_c = Complexity()
    yngve_mean, frazier_mean, sdl_mean = syntactic_c.syntactic_complexity(parser_output)

    # Calculating lexical complexity
    lexical_c = LexicalComplexity()
    print("Passing to lexical ",refined_text)
    ttr_lematized, ttr, honore_statistics, ARI, brunet_index, CLI = lexical_c.get_lexical_measures(refined_text)


    measures_type = [
        'Syntactic Measures',
        'Yngve_mean',
        'Frazier_mean',
        'Mean Syntactic Dependency Length',
        '',
        'Lexical Measures',
        'Type to token ratio',
        'Type to token ratio: Lemmatized text',
        'honore_statistics',
        'Automatic Readability Index',
        'brunet_index',
        'Coleman Liau\'s index'
        ''
    ]

    measures_value = [
        '', 
        str(yngve_mean),
        str(frazier_mean),
        str(sdl_mean),
        '',
        '',
        str(ttr),
        str(ttr_lematized),
        str(honore_statistics),
        str(ARI),
        str(brunet_index),
        str(CLI)
    ]

    f = open("Complexity Measures.csv", "w")
    f.write("{},{}\n".format("Complexity Measures", "Values"))
    for x in zip(measures_type, measures_value):
        f.write("{},{}\n".format(x[0], x[1]))
    f.close()

if __name__ == '__main__':
    main()
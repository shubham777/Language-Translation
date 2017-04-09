# -*- coding: utf-8 -*-
import re


class Tokenizer(object):
    def __init__(self):
        pass

        self.abbreviations = {'dr.': 'doctor', 'mr.': 'mister', 'bro.': 'brother', 'bro': 'brother', 'mrs.': 'mistress',
                         'ms.': 'miss', 'jr.': 'junior', 'sr.': 'senior',
                         'i.e.': 'for example', 'e.g.': 'for example', 'vs.': 'versus'}
        self.terminators = ['.', '!', '?']
        self.wrappers = ['"', "'", ')', ']', '}']


        # starting quotes
        self.STARTING_QUOTES = [
            (re.compile(r'^\"'), r'``'),
            (re.compile(r'(``)'), r' \1 '),
            (re.compile(r'([ (\[{<])"'), r'\1 `` '),
        ]

        # punctuation
        self.PUNCTUATION = [
            (re.compile(r'([:,])([^\d])'), r' \1 \2'),
            (re.compile(r'([:,])$'), r' \1 '),
            (re.compile(r'\.\.\.'), r' ... '),
            (re.compile(r'[;@#$%&]'), r' \g<0> '),
            (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
            (re.compile(r'[?!]'), r' \g<0> '),

            (re.compile(r"([^'])' "), r"\1 ' "),
        ]

        # parens, brackets, etc.
        self.PARENS_BRACKETS = [
            (re.compile(r'[\]\[\(\)\{\}\<\>]'), r' \g<0> '),
            (re.compile(r'--'), r' -- '),
        ]

        # ending quotes
        self.ENDING_QUOTES = [
            (re.compile(r'"'), " '' "),
            (re.compile(r'(\S)(\'\')'), r'\1 \2 '),

            (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
            (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
        ]

        # List of contractions adapted from Robert MacIntyre's tokenizer.
        self.CONTRACTIONS2 = [re.compile(r"(?i)\b(can)(not)\b"),
                         re.compile(r"(?i)\b(d)('ye)\b"),
                         re.compile(r"(?i)\b(gim)(me)\b"),
                         re.compile(r"(?i)\b(gon)(na)\b"),
                         re.compile(r"(?i)\b(got)(ta)\b"),
                         re.compile(r"(?i)\b(lem)(me)\b"),
                         re.compile(r"(?i)\b(mor)('n)\b"),
                         re.compile(r"(?i)\b(wan)(na) ")]
        self.CONTRACTIONS3 = [re.compile(r"(?i) ('t)(is)\b"),
                         re.compile(r"(?i) ('t)(was)\b")]
        self.CONTRACTIONS4 = [re.compile(r"(?i)\b(whad)(dd)(ya)\b"),
                         re.compile(r"(?i)\b(wha)(t)(cha)\b")]




    def tokenize(self, text):

        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        for regexp, substitution in self.PARENS_BRACKETS:
            text = regexp.sub(substitution, text)

        # add extra space to make things easier
        text = " " + text + " "

        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r' \1 \2 ', text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r' \1 \2 ', text)

        # We are not using CONTRACTIONS4 since
        # they are also commented out in the SED scripts
        # for regexp in self.CONTRACTIONS4:
        #     text = regexp.sub(r' \1 \2 \3 ', text)

        return text.split()


    def get_sentences(self, text):
        end = True
        sentences = []
        while end > -1:
            end = self.find_sentence_end(text)
            if end > -1:
                sentences.append(text[end:].strip())
                text = text[:end]
        sentences.append(text)
        sentences.reverse()
        return sentences

    def find_sentence_end(self, paragraph):

        [possible_endings, contraction_locations] = [[], []]
        contractions = self.abbreviations.keys()
        sentence_terminators = self.terminators + [terminator + wrapper for wrapper in self.wrappers for terminator in
                                                   self.terminators]

        for sentence_terminator in sentence_terminators:
            t_indices = list(self.find_all(paragraph, sentence_terminator))
            possible_endings.extend(([] if not len(t_indices) else [[i, len(sentence_terminator)] for i in t_indices]))

        for contraction in contractions:
            c_indices = list(self.find_all(paragraph, contraction))
            contraction_locations.extend(([] if not len(c_indices) else [i + len(contraction) for i in c_indices]))

        possible_endings = [pe for pe in possible_endings if pe[0] + pe[1] not in contraction_locations]

        if len(paragraph) in [pe[0] + pe[1] for pe in possible_endings]:
            max_end_start = max([pe[0] for pe in possible_endings])
            possible_endings = [pe for pe in possible_endings if pe[0] != max_end_start]

        possible_endings = [pe[0] + pe[1] for pe in possible_endings if
                            sum(pe) > len(paragraph) or (sum(pe) < len(paragraph) and paragraph[sum(pe)] == ' ')]
        end = (-1 if not len(possible_endings) else max(possible_endings))
        return end

    def find_all(self, a_str, sub):
        start = 0
        while True:
            start = a_str.find(sub, start)
            if start == -1:
                return
            yield start
            start += len(sub)



t = Tokenizer()
text = u"I am not criticising this; it happens from time to time that people send someone to represent them."
sent = t.get_sentences(text)

tokenized_sentence = t.tokenize(sent[0])
pass


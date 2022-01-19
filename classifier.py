import re
import json
from nltk.util import ngrams


class RuleClassifier:
    def __init__(self, data, kb_path):
        self.data = data
        self.lexicon = None
        self.patterns = None
        self.res = []

        with open(kb_path, 'r') as f:
            kb = json.load(f)
            self.lexicon = kb['LEXICON']
            self.pattern = kb['PATTERNS']

    @staticmethod
    def ngram_filter(record, word, n):
        tokens = record.split()
        all_ngrams = ngrams(tokens, n)
        filtered_ngrams = [x for x in all_ngrams if word in x]
        return filtered_ngrams

    def get_ngrams(self, text):
        smoke_reg = re.search('smok\\w+', text)
        if smoke_reg:
            smoke_word = smoke_reg.group(0)
            ngrams = self.ngram_filter(text, smoke_word, 4)
            return ngrams

    def tag_from_context(self, ngrams):
        label = 'Unknown'
        try:
            for ngram in ngrams:
                if any(item in self.lexicon['former'] for item in ngram):
                    label = 'Former Smoker'
                    break
                elif re.search(self.pattern['is_past'], ' '.join(ngram)):
                    label = 'Former Smoker'
                    break
                elif any(item in self.lexicon['ongoing'] for item in ngram):
                    label = 'Smoker'
                    break
                elif re.search(self.pattern['is_present'], ' '.join(ngram)):
                    label = 'Smoker'
                    break
                elif re.search(r'smoked|did smoke', ' '.join(ngram)):
                    label = 'Former Smoker'
                    break
                elif re.search(r'smokes', ' '.join(ngram)):
                    label = 'Smoker'
                    break
                else:
                    label = 'Unknown'
        except TypeError as e:
            print(e)

        return label

    def classify(self, text):
        if re.search(r'nonsmoker|does not (\w+\\s)?smoke', text):
            self.res.append(('Non Smoker', text))
        elif re.search(r'to quit', text):
            self.res.append(('Smoker', text))
        elif re.search(r'(?<!to\\s)quit|former', text):
            self.res.append(('Former Smoker', text))
        else:
            ngrams = self.get_ngrams(text)
            label = self.tag_from_context(ngrams)
            self.res.append((label, text))
        return

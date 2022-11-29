import predictor
from text_mutator import TextMutator
from text_input import Text

class Individual(object):
    # Global counter of all the individuals (it is increased each time an individual is created or mutated).
    COUNT = 0
    SEEDS = set()
    COUNT_MISS = 0

    def __init__(self, text: Text):
        self.member = text
        self.features = tuple()
        self.ff = None

    def reset(self):
        self.ff = None

    def evaluate(self):
        if self.ff is None: 
            self.member.predicted_label, self.member.confidence = predictor.Predictor.predict(self.member.text)   

            self.ff = self.member.confidence if self.member.confidence > 0 else -0.1      
           
        return self.ff

    def mutate(self):
        TextMutator(self.member).mutate()
        self.reset()


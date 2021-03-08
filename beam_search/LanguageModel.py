class LanguageModel:

    def __init__(self, lm, classes):
        self.initCharBigrams(lm, classes)

    def initCharBigrams(self, lm, classes):
        "internal init of character bigrams"

        # init bigrams with 0 values
        self.bigram = {c: {d: 0 for d in classes} for c in classes}

        # go through text and add each char bigram
        txt = lm
        for i in range(len(txt) - 1):
            first = txt[i]
            second = txt[i + 1]

            # ignore unknown chars
            if first not in self.bigram or second not in self.bigram[first]:
                continue

            self.bigram[first][second] += 1

    def getCharBigram(self, first, second):
        "probability of seeing character 'first' next to 'second'"
        first = first if first else ' '  # map start to word beginning
        second = second if second else ' '  # map end to word end

        # number of bigrams starting with given char
        numBigrams = sum(self.bigram[first].values())
        if numBigrams == 0:
            return 0
        return self.bigram[first][second] / numBigrams

import numpy as np
import nltk
import random, glob, pickle


class ParettoDataset:
    def __init__(self, ori_data, class_min_count=50, token_min_count=50, split=np.array([0.6, 0.8, 1.0]), tablesize=1000000):
        self.class_min_count = class_min_count
        self.token_min_count = token_min_count
        self.tablesize = tablesize

        self.data = self.preprocess_data(ori_data)

        random.shuffle(self.data)

        self.split_tr = int(len(self.data) * split[0])
        self.split_va = int(len(self.data) * split[1])
        self.split_tx = int(len(self.data) * split[2])

    def preprocess_data(self, ori_data):
        unique_ac = list(set(ori_data['ac']))

        ac_list_useful = []
        for iAc in unique_ac:
            if ori_data.ac[ori_data.ac==iAc].count() > self.class_min_count:
                ac_list_useful.append(iAc)

        self.useful_ac = ac_list_useful

        print '[Info]: Unique ac count: ', len(unique_ac)
        print '[Info]: Selected useful ac count: ', len(ac_list_useful)

        tr_tx_data = []
        for (linenum, line) in ori_data.iterrows():
            if line['ac'] in ac_list_useful:
                    tr_tx_data.append((line['r'], line['ac']))

        return tr_tx_data

    def getTrainData(self):
        return self.data[:self.split_tr]

    def getValidData(self):
        return self.data[self.split_tr:self.split_va]

    def getTestData(self):
        return self.data[self.split_va:self.split_tx]

    def getTokens(self, load_prev, prev_tokens=None):
        if hasattr(self, "tokens") and self.tokens:
            return self.tokens

        token_data = self.getTrainData()

        self.r_sentence = []
        all_tokens = []
        for (r, ac) in token_data:
            tokens = nltk.word_tokenize(r)
            all_tokens += tokens
            self.r_sentence.append(tokens)

        self.tot_word_count = len(all_tokens)

        fDict = nltk.FreqDist(all_tokens)
        self.freqDict = fDict

        tokens_list = list(set(all_tokens))

        if load_prev:
            self.tokens = prev_tokens
        else:
            self.tokens = dict()
            for i in range(len(tokens_list)):
                self.tokens[tokens_list[i]] = i

        self.tokens_list = []
        for iDict in self.tokens:
            self.tokens_list.append(iDict)

        self.nTokens = len(self.tokens)
        print '[Info]: Word tokens count: ', self.nTokens

        return self.tokens

    def getFreqTokens(self):
        freq_token = []
        for iWord in self.freqDict.items():
                if iWord[1] > self.token_min_count:
                    freq_token.append(iWord[0])

        for (word, tag) in nltk.pos_tag(freq_token):
            if tag == 'DT' or tag == 'IN' or tag == 'TO' or tag == 'MD' or tag == 'CC' or tag == 'CD':
                freq_token.remove(word)

        return freq_token

    def getRejectProb(self):
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb

        threshold = self.tot_word_count * 1e-2
        rejectProb = np.zeros((self.nTokens, ))

        for i in xrange(self.nTokens):
            w = self.tokens_list[i]
            if w in self.freqDict:
                freq = 1.0 * self.freqDict[w]
                # Reweigh
                rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq))
            else:
                rejectProb[i] = 0

        self._rejectProb = rejectProb
        return self._rejectProb

    def getAllSentences(self):
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        rejectProb = self.getRejectProb()

        allsentences = [[w
            for w in s if w in self.tokens
            if 0 >= rejectProb[self.tokens[w]] or random.random() >= rejectProb[self.tokens[w]]]
            for s in self.r_sentence * 30]

        allsentences = [s for s in allsentences if len(s) > 1]

        self._allsentences = allsentences
        return self._allsentences

    def getRandomContext(self, C=5):
        allsent = self.getAllSentences()
        sentID = random.randint(0, len(allsent) - 1)
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1)

        context = sent[max(0, wordID - C):wordID]
        if wordID+1 < len(sent):
            context += sent[wordID+1:min(len(sent), wordID + C + 1)]

        centerword = sent[wordID]
        context = [w for w in context if w != centerword and w in self.tokens]

        if len(context) > 0 and centerword in self.tokens:
            return centerword, context
        else:
            return self.getRandomContext(C)

    def sampleTable(self):
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable

        samplingFreq = np.zeros((self.nTokens,))

        i = 0
        for w in xrange(self.nTokens):
            w = self.tokens_list[i]
            if w in self.freqDict:
                freq = 1.0 * self.freqDict[w]
                # Reweigh
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1

        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize

        self._sampleTable = [0] * self.tablesize

        j = 0
        for i in xrange(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j

        return self._sampleTable

    def sampleTokenIdx(self):
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]

    def getClassVec(self, word_vec):
        tr_data = self.getTrainData()

        class_vec = np.zeros((len(self.useful_ac), word_vec.shape[1]))
        class_word_count = np.zeros((len(self.useful_ac), 1))

        useful_ac_dict = dict()
        for i in range(len(self.useful_ac)):
            useful_ac_dict[self.useful_ac[i]] = i

        for (r, ac) in tr_data:
            idx = useful_ac_dict[ac]
            for w in r:
                try:
                    class_vec[idx, :] += word_vec[self.tokens[w], :]
                    class_word_count[idx, :] += 1
                except:
                    pass

        return self.useful_ac, class_vec / class_word_count


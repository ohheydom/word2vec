from os import walk
import string

class CorpusReader:
    """ CorpusReader takes in a folder name and provides helper methods
    for the word2vec implementation. 
    """

    def __init__(self, folder, window_size=1):
        self.folder = folder
        self.window_size = window_size
        self.dictionary = {}
        self.location = 0
        self.text = []

    def build_dictionary(self):
        """ build_dictionary Builds the dictionary of all the words from 
        a given folder. Also builds an array of the entire text corpus.

        Returns
        -------
        int : Length of the dictionary
        """

        for root, directory, files in walk(self.folder):
            for f in files:
                with open('{}/{}'.format(root, f)) as of:
                    for line in of:
                        for word in line.split():
                            w = word.lower().strip(string.punctuation)
                            self.text.append(w)
                            if w not in self.dictionary:
                                self.dictionary[w] = len(self.dictionary)
        return len(self.dictionary)


    def next_batch(self, batch_size=10):
        """ next_batch Returns a 2 dimensional list of length batch_size like so:
        [[word, context], [word, context], [word, context]]
        The words and contexts are the indices of the words from
        the dictionary mapping.

        Returns
        -------
        list : shape(batch_size, 2)
        """

        l = []
        text_len = len(self.text)
        for _ in self.text:
            window = 1
            idx = self.location
            self.location += 1
            if idx >= text_len:
                self.location = 0
                continue
            while window <= self.window_size and idx-window >= 0:
                l.append([self.dictionary[self.text[idx]], self.dictionary[self.text[idx-window]]])
                if len(l) == batch_size:
                    return l
                window += 1
            window = 1
            while window <= self.window_size and idx+window <= text_len-1:
                l.append([self.dictionary[self.text[idx]], self.dictionary[self.text[idx+window]]])
                if len(l) == batch_size:
                    return l
                window += 1
        return l

import sys

class PartialParse(object):
    def __init__(self, sentence):
        """
        @param sentence (list of str): The sentence to be parsed as a list of words.
                                      
        """
        self.sentence = sentence

        self.stack = ['ROOT']
        self.buffer = sentence[:]
        self.dependencies = []
        


    def parse_step(self, transition):
        """
        @param transition (str): A string that equals "S", "LA", or "RA" representing the shift,
                                left-arc, and right-arc transitions. 
        """

        if transition is 'S':
            word = self.buffer[0]
            self.stack.append(word)
            self.buffer.remove(word)

        elif transition is 'LA':
            l = len(self.stack)
            head = self.stack[l-1]
            dependent = self.stack[l-2]
            self.stack.remove(dependent)
            self.dependencies.append((head,dependent))

        elif transition is 'RA':
            l = len(self.stack)
            head = self.stack[l-2]
            dependent = self.stack[l-1]
            self.stack.remove(dependent)
            self.dependencies.append((head,dependent))


    def parse(self, transitions):
        """
        @param transitions (list of str): The list of transitions in the order they should be applied

        @return dependencies (list of string tuples): The list of dependencies produced when
                                                        parsing the sentence. Represented as a list of
                                                        tuples where each tuple is of the form (head, dependent).
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    """
    @param sentences (list of list of str): A list of sentences to be parsed
                                            
    @param model (ParserModel): The model that makes parsing decisions. 

    @param batch_size (int): The number of PartialParses to include in each minibatch

    @return dependencies (list of dependency lists): A list where each element is the dependencies
                                                    list for a parsed sentence. 
    """
    dependencies = []
    
    partial_parses = [PartialParse(sentence) for sentence in sentences]
    
    unfinished_parses = partial_parses[:]

    while unfinished_parses:

        minibatch = unfinished_parses[0:batch_size]
        transitions = model.predict(minibatch)
        
        for transition, partial_parse in zip(transitions, minibatch):
            partial_parse.parse_step(transition)

            if len(partial_parse.buffer) == 0 and len(partial_parse.stack) == 1:
                unfinished_parses.remove(partial_parse) 
            

    dependencies = [parse.dependencies for parse in partial_parses] 

    return dependencies



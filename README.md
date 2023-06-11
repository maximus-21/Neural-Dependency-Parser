# Neural-Dependency-Parser
This is an implementation of Transition Based Dependency Parser Using Neural Networks. This implementation is based on one of the assignments from [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)

A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between head words, and words which modify those heads. There are multiple types of dependency parsers, including transition-based parsers, graph-based parsers, and feature-based parsers. Our implementation will be a transition-based parser, which incrementally builds up a parse one step at a time. At every step it maintains a partial parse, which is represented as follows:
- A stack of words that are currently being processed.
- A buffer of words yet to be processed.
- A list of dependencies predicted by the parser.

Initially, the stack only contains ROOT, the dependencies list is empty, and the buffer contains all words of the sentence in order. At each step, the parser applies a transition to the partial parse until its buffer is empty and the stack size is 1. The following transitions can be applied:
- SHIFT: removes the first word from the buffer and pushes it onto the stack.
- LEFT-ARC: marks the second (second most recently added) item on the stack as a dependent of
the first item and removes the second item from the stack, adding a first word → second word
dependency to the dependency list.
- RIGHT-ARC: marks the first (most recently added) item on the stack as a dependent of the second
item and removes the first item from the stack, adding a second word → first word dependency to
the dependency list.

A neural network predicts which of the 3 transitions (SHIFT, LEFT-ARC or RIGHT-ARC) the parser needs to apply in its next step, given the state of the stack, buffer and the dependencies. 
- First, the model extracts a feature vector representing the current state. We will be using the feature set presented in the original neural dependency parsing paper: [A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf). This feature vector consists of a list of tokens (e.g., the last word in the stack, first word in the buffer, dependent of the second-to-last word in the stack if there is one, etc.). 
 - The input to the neural network is a concatenated vector of the embedding for each of these tokens.
- The final output layer has 3 nodes (SHIFT, LEFT-ARC and RIGHT-ARC)


**parser_model.py** : Model for the neural network classifier.

**parser_transitions** : For implementing transitions to the parsers.

**run.py** : For Training and Testing Model


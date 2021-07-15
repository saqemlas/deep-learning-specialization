# Week 4 Quiz - Transformers

1. A Transformer Network, like its predecessors RNNs, GRUs and LSTMs, can process information one word at a time. (Sequential architecture).

    [] True

    [X] False

2. Transformer Network methodology is taken from: (Check all that apply)

    [] Convolutional Neural Network style of processing.

    [] None of these.

    [X] Attention mechanism.

    [X] Convolutional Neural Network style of architecture.

3. The concept of Self-Attention is that: 

    [X] Given a word, its neighbouring words are used to compute its context by summing up the word values to map the Attention related to that given word.

    [] Given a word, its neighbouring words are used to compute its context by selecting the highest of those word values to map the Attention related to that given word.

    [] Given a word, its neighbouring words are used to compute its context by taking the average of those word values to map the Attention related to that given word.

    [] Given a word, its neighbouring words are used to compute its context by selecting the lowest of those word values to map the Attention related to that given word.

4. Which of the following correctly represents Attention ?

    [X] softmax... V

    [] 

    []

    [] 

5. Are the following statements true regarding Query (Q), Key (K) and Value (V) ?

Q = interesting questions about the words in a sentence

K = specific representations of words given a Q

V = qualities of words given a Q

    [] True

    [X] False

6. i here represents the computed attention weight matrix associated with the ithith “word” in a sentence.

    [] True

    [X] False

7. Following is the architecture within a Transformer Network. (without displaying positional encoding and output layers(s)) What information does the Decodertake from the Encoder for its second block of Multi-Head Attention ? (Marked XX, pointed by the independent arrow) (Check all that apply)

    [X] K

    [X] V

    [] Q

8. Following is the architecture within a Transformer Network. (without displaying positional encoding and output layers(s)) What is the output layer(s) of the Decoder ? (Marked YY, pointed by the independent arrow)

    [X] Linear layer followed by a softmax layer.

    [] Softmax layer followed by a linear layer.

    [] Linear layer

    [] Softmax layer

9. Why is positional encoding important in the translation process? (Check all that apply)

    [X] Position and word order are essential in sentence construction of any language.

    [] It helps to locate every word within a sentence.

    [] It is used in CNN and works well there.

    [X] Providing extra information to our model.

10. Which of these is a good criteria for a good positionial encoding algorithm?

    [X] It should output a unique encoding for each time-step (word’s position in a sentence).

    [X] Distance between any two time-steps should be consistent for all sentence lengths.

    [X] The algorithm should be able to generalize to longer sentences.

    [] None of the these.

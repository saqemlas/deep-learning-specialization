# Week 2 Quiz - Natural Language Processing & Word Embeddings

1. Suppose you learn a word embedding for a vocabulary of 10000 words. Then the embedding vectors should be 10000 dimensional, so as to capture the full range of variation and meaning in those words.

    [] True

    [X] False

2. What is t-SNE?

    [] A linear transformation that allows us to solve analogies on word vectors

    [X] A non-linear dimensionality reduction technique

    [] A supervised learning algorithm for learning word embeddings

    [] An open-source sequence modeling library

3. Suppose you download a pre-trained word embedding which has been trained on a huge corpus of text. You then use this word embedding to train an RNN for a language task of recognizing if someone is happy from a short snippet of text, using a small training set. Then even if the word “ecstatic” does not appear in your small training set, your RNN might reasonably be expected to recognize “I’m ecstatic” as deserving a label y = 1y=1.

    [X] True

    [] False

4. Which of these equations do you think should hold for a good word embedding? (Check all that apply) 

    [] eboy - egirl ~~ esister - ebrother

    [X] eboy - ebrother ~~ egirl - esister

    [X] eboy - egirl ~~ ebrother - esister

    [] eboy - ebrother ~~ esister - egirl

5. Let E be an embedding matrix, and let {O}1234 be a one-hot vector corresponding to word 1234. Then to get the embedding of word 1234, why don’t we call E * {O}1234 in Python?

    [] None of the above: calling the Python snippet as described above is fine.

    [] This doesn’t handle unknown words (<UNK>).

    [] The correct formula is E^t * {O}1234

    [X] It is computationally wasteful.

6. When learning word embeddings, we create an artificial task of estimating P(target \mid context)P(target∣context). It is okay if we do poorly on this artificial prediction task; the more important by-product of this task is that we learn a useful set of word embeddings.

    [X] True

    [] False

7. In the word2vec algorithm, you estimate P(t∣c), where tt is the target word and cc is a context word. How are tt and cc chosen from the training set? Pick the best answer.

    [] c is a sequence of several words immediately before t.

    [] c is the one word that comes immediately before t.

    [] c is the sequence of all the words in the sentence before t.

    [X] c and t are chosen to be nearby words.

8. Suppose you have a 10000 word vocabulary, and are learning 500-dimensional word embeddings. The word2vec model uses the following softmax function: Which of these statements are correct? Check all that apply.

    [X] Ot and Ec are both 500 dimensional vectors. 

    [X] Ot and Ec are both trained with an optimization algorithm such as Adam or gradient descent. 

    [] Ot and Ec are both 10000 dimensional vectors.

    [] After training, we should expect Ot to be very close to Ec when tt and cc are the same word. 

9. Suppose you have a 10000 word vocabulary, and are learning 500-dimensional word embeddings.The GloVe model minimizes this objective: Which of these statements are correct? Check all that apply.

    [X] Xij is the number of times word j appears in the context of word i.

    [X] Oi and Ej should be initialized randomly at the beginning of training.

    [X] The weighting function f(.) must satisfy f(O) = O.

    [] Oi and Ej should be initialized to 0 at the beginning of training.

10. You have trained word embeddings using a text dataset of M1 words. You are considering using these word embeddings for a language task, for which you have a separate labeled dataset of M2 words. Keeping in mind that using word embeddings is a form of transfer learning, under which of these circumstances would you expect the word embeddings to be helpful?

    [X] M1 >> M2

    [] M1 << M2


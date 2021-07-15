# Week 3 Quiz - Sequence Models & Attention Mechanism

1. Consider using this encoder-decoder model for machine translation. This model is a “conditional language model” in the sense that the encoder portion (shown in green) is modeling the probability of the input sentence x.

    [X] True

    [] False

2. In beam search, if you increase the beam width BB, which of the following would you expect to be true? Check all that apply.

    [X] Beam search will generally find better solutions (i.e. do a better job maximizing P(y \mid x)P(y∣x))

    [X] Beam search will use up more memory.

    [] Beam search will converge after fewer steps. 

    [X] Beam search will run more slowly.

3. In machine translation, if we carry out beam search without using sentence normalization, the algorithm will tend to output overly short translations. 

    [X] True

    [] False 

4. Suppose you are building a speech recognition system, which uses an RNN model to map from audio clip x to a text transcript y. Your algorithm uses beam search to try to find the value of y that maximizes P(y∣x). On a dev set example, given an input audio clip, your algorithm outputs the transcript y = “I’m building an A Eye system in Silly con Valley.”, whereas a human gives a much superior transcript y = “I’m building an AI system in Silicon Valley.” According to your model,
    P(y|x)=1.09*10-7
    P(y*|x)=7.21*10-8

Would you expect increasing the beam width B to help correct this example?

    [] No, because P(y*|x)<P(y|x) indicates the error should be attributed to the search algorithm rather than to the RNN.

    [X] No, because P(y*|x)<P(y|x) indicates the error should be attributed to the RNN rather than to the search algorithm.

    [] Yes, because P(y*|x)<P(y|x) indicates the error should be attributed to the search algorithm rather than to the RNN.

    [] Yes, because P(y*|x)<P(y|x) indicates the error should be attributed to the RNN rather than to the search algorithm.

5. Continuing the example from Q4, suppose you work on your algorithm for a few more weeks, and now find that for the vast majority of examples on which your algorithm makes a mistake, P(y*|x)>P(y|x) This suggests you should focus your attention on improving the search algorithm.

    [X] True

    [] False

6. Consider the attention model for machine translation. Further, here is the formula for a^<t,t>. Which of the following statements about a^<t,t>. are true? Check all that apply.

    [] We expect a^<t,t'> to be generally larger for values of a^<t> that are highly relevant to the value the network shoud output for y^<t'>

    [X] We expect a^<t,t'> to be generally larger for values of a^<t'> that are highly relevant to the value the network shoud output for y^<t>

    [] Eta a^<t,t> = 1

    [X] Et'a a^<t,t> = 1

7. The network learns where to “pay attention” by learning the values e^<t,t'>, which are computed using a small neural network: We can't replace s^<t-1> with s^<t> as an input to this neural network. This is because s^<t> depends on a^<t,t'> which in turn depends on e^<t,t'>; so at the time we need to evaluate this network, we haven't compute s^<t> yet.

    [X] True

    [] False

8. Compared to the encoder-decoder model shown in Question 1 of this quiz (which does not use an attention mechanism), we expect the attention model to have the greatest advantage when:

    [X] The input sequence length Tx is large.

    [] Ths input sequence legnth Tx is small.

9. Under the CTC model, identical repeated characters not separated by the “blank” character (_) are collapsed. Under the CTC model, what does the following string collapse to?

__c_oo_o_kk___b_ooooo__oo__kkk

    [] coookkboooooookkk

    [X] cookbook

    [] cokbok

    [] cook book

10. In trigger word detection, x^<t> is:

    [X] Features of the audio (such as spectrogram features) at time tt.

    [] The tt-th input word, represented as either a one-hot vector or a word embedding.

    [] Whether the trigger word is being said at time tt.

    [] Whether someone has just finished saying the trigger word at time tt.


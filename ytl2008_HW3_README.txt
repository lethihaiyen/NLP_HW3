NLP Homework 3
Yen Le 
* I was granted a late submisison by the time I submit this homework. Thank you very much ! * 

This submission contains a Python file that runs the Viterbi HMM POS tagger system.
 The Python file has functions used for processing our files, calculating probabilities table needed for the algorithm to run. 

To run the system, just go on commans line and run the command : python3 main_ytl2008_HW3.py
The system is build in Python and will output a submission file "submission.pos" in the same directory. 

I implemented the algorithm as shown in class, and tweek my specification in OOV function to improve the accuracy of my work. 
There are serveral thing I did to improve tagging unknown words ( can be found in  OOV function)
(1) Hard-coding : 
    - Specific word : "the" (DT) ,"by"(IN) ,"to"(TO)
    - Punctuation/ Open_Close tags are of a POS of itself. Eg, "," is "," with probability 1 
    - Number are all regards as CD 
    - Currency all have POS 
    - Capitalized words are NNP 
    - Words start with Capitalized letter are either NNP or NNN ( though I knowledge this might be restricting)
    - Words ends with s are predicted to be either NNS, VBZ, NNP, or NNPS
    - Words ending in "ish", "ous", "ful", "less", "ble", "ive", "us" were predicted to be JJ
(2) Others are assigned probability of 1/1000 for every tag. 

# One note : 
- I deal with this bug for a day and still haven't figure out where I was wrong, but I eventually figure it out : 
- In the transional table and likelihood table, I put the probability of an unseen transition as 0. This has greatly drive my POS tagging
badly ( worsen by 7% accuracy ). So I change to a softer margin, establishing every case that I haven't seen with probability 1e^-10. This 
will allow my model to be able to make better prediction and have more flexibility when it comes to evaluating POS tag possibilities. 



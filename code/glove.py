# -*- coding: utf-8 -*-
# @Time       : 2021/03/08 16:12:52
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : projects
# @Description:  

import dill
import argparse
import pdb
import logging

import numpy as np
from tqdm import tqdm
from random import shuffle
from math import log

logger = logging.getLogger("glove")
epsilon = 0.01

def run_iter(data,learning_rate = 0.05,x_max =100,alpha = 0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.
    `data` is a pre-fetched data / weights list where each element is of
    the form
        (v_main, v_context,
         b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)
    as produced by the `train_glove` function. Each element in this
    tuple is an `ndarray` view into the data structure which contains
    it.
    See the `train_glove` function for information on the shapes of `W`,
    `biases`, `gradient_squared`, `gradient_squared_biases` and how they
    should be initialized.
    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; see the GloVe paper for more
    details.
    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.
    """
    global_cost = 0

    shuffle(data)

    for(v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
        gradsq_b_main, gradsq_b_context, cooccurrence) in data:

        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # Compute inner component of cost function, which is used in
        # both overall cost calculation and in gradient calculation
        # $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij})$$
        # pdb.set_trace()
        cost_inner = v_main.dot(v_context) + b_main[0] + b_context[0] - log(cooccurrence)

        # Compute cost
        # $$ J = f(X_{ij})(J')^2 $$
        cost = weight *(cost_inner ** 2)

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost

        # Compute gradient for word vector terms
        # derivation of w_i 
        #   grad(w_i) = 2 * f(X_{ij})* w_j * J'
        #   derivation of b_i
        #   grad(b_i) = 2 * f(X_{ij})* J'
        grad_main = weight * cost_inner * v_context
        grad_context = weight * cost_inner * v_main

        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner
        
        # perform adaptive updates
        v_main -= learning_rate * grad_main / np.sqrt(gradsq_W_main)
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
                gradsq_b_context))

         # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2

        return global_cost
        
def train_glove(vocab,cooccurrences,vector_size = 100,iterations = 1000):
    """
    train glove vector on the given 'coocurrences',where 
    each element of the form is 
        map : (word_i_id,word_j_id): x_ij
    Return the word vector matrix W
    """
    vocab_size = len([idx for idx in vocab])
    # Word vector matrix. This matrix is (2N) * d, where N is the size
    # of the corpus vocabulary and d is the dimensionality of the word
    # vectors. All elements are initialized randomly in the range (-0.5,
    # 0.5]. We build two word vectors for each word: one for the word as
    # the main (center) word and one for the word as a context word.
    #
    # It is up to the client to decide what to do with the resulting two
    # vectors. Pennington et al. (2014) suggest adding or averaging the
    # two for each word, or discarding the context vectors.
    # TODO: 源代码中还除以了一个float(vector_size + 1)，不知道啥意思
    # pdb.set_trace()
    W = (np.random.rand(vocab_size * 2,vector_size)-0.5)/ float(vector_size + 1)
    biases = (np.random.rand(vector_size*2) - 0.5)/ float(vector_size + 1)

    # Training is done via adaptive gradient descent (AdaGrad). To make
    # this work we need to store the sum of squares of all previous
    # gradients.
    #
    # Like `W`, this matrix is (2V) * d.
    #
    # Initialize all squared gradient sums to 1 so that our initial
    # adaptive learning rate is simply the global learning rate.
    gradient_squared = np.ones((vocab_size * 2, vector_size),
                               dtype=np.float64)
    # Sum of squared gradients for the bias terms.
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    data = [(W[i_main], W[i_context + vocab_size],
             biases[i_main : i_main + 1],
             biases[i_context + vocab_size : i_context + vocab_size + 1],
             gradient_squared[i_main], gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_main : i_main + 1],
             gradient_squared_biases[i_context + vocab_size
                                     : i_context + vocab_size + 1],
             cooccurrence)   for (i_main, i_context),cooccurrence in cooccurrences.items()]  

    for i in tqdm(range(iterations)):
        cost = run_iter(data)

        logger.info(f"\t\tTime {i} Done (cost %f)", cost)   

        log_cost('Time ' +str(i) +' done, cost '+ str(cost)+'\n')

        if(cost <= epsilon):
            break
    
    save_model(W)

    return W

def save_model(W, path="../data/"):
    with open("../data/" + "embedding.pkl", 'wb') as f:
        dill.dump(W, f)

    logger.info("Saved vectors to %s", path + "embedding.pkl")

def log_cost(data):
    # pdb.set_trace()
    with open("../" + "log.txt","a") as f:
        f.write(data)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s\t%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",default = "../data/",help = "the path that stores data")
    parser.add_argument("--coMap",default = "cooccurrenceMap.pkl",help = "cooccurencemap")
    parser.add_argument("--voc",default = "type.pkl",help = "the voc file")
    args = parser.parse_args()	    

    cooccurrences = dill.load(open(args.data_path + args.coMap,"rb"))
    # vocab = dill.load(open(args.data_path + args.voc,"rb"))['diag_voc'].idx2word
    vocab = dill.load(open(args.data_path + args.voc,"rb"))
    # pdb.set_trace()
    train_glove(vocab,cooccurrences,len(vocab))
import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    Base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=self.verbose).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ 
	Select the model with value self.n_constant
    """

    def select(self):
        """ 
		Select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ 
	Select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ 
		Select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_bic = float("+inf")
        best_model = None

        N = len(self.X)
        num_features = len(self.X[0])

        for num_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_components)
                logL = model.score(self.X, self.lengths)              
                transition_probs = num_components * (num_components - 1)
                starting_probs = num_components - 1
                n_means = num_components * num_features
                n_variances = num_components * num_features

                p = transition_probs + starting_probs + n_means + n_variances

				# BIC
                bic = -2 * logL + p * math.log(N)

                if  bic < min_bic:
                    min_bic = bic
                    best_model = model
            except:
                continue
        return best_model


class SelectorDIC(ModelSelector):
    ''' 
	Select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        ModelSelector.__init__(self, all_word_sequences, all_word_Xlengths, this_word,
                 n_constant, min_n_components, max_n_components,random_state, verbose)

        self.other_words = [word for word in self.hwords if word != self.this_word]

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_dic = float("-inf")
        best_model = None
        M = len(self.hwords)

        for num_components in range(self.min_n_components,self.max_n_components + 1):
            try:
                model = self.base_model(num_components)
                logL = model.score(self.X, self.lengths)

                logL_other_words = [model.score(X, lengths) for (X, lengths) in [self.hwords[word] for word in self.other_words]]

				# DIC
                dic = logL - np.average(logL_other_words)

                if dic > max_dic:
                    max_dic = dic
                    best_model = model
            except Exception as e:
                if self.verbose:
                    print("Error for {} components: {}".format(num_components, str(e)))

        return best_model
		

class SelectorCV(ModelSelector):
    ''' 
    Select best model based on average log Likelihood of cross-validation folds
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_score = float('-inf')

        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                split_method = KFold(n_splits=min(3,len(self.lengths)))
                logL = []
                for train_id, test_id in split_method.split(self.sequences):
                    train_n,train_len = combine_sequences(train_id, self.sequences)
                    test_n,test_len = combine_sequences(test_id, self.sequences)
                    # training model
                    current_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(train_n, train_len)
                    # evaluating model
                    logL.append(current_model.score(test_n, test_len))
                    
                current_score = np.average(logL)
                
                if current_score > best_score:
                    best_score = current_score
                    best_model = current_model
            except:
                pass

        return best_model
import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
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
        """
        Universal model selector function for select() methods of SelectorCV, SelectorBIC and SelectorDIC
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        #print('Inside base select() function. Sequences passed:', self.sequences)
        best_score = -float('inf')
        best_components_number = self.min_n_components
        best_model = None
        n_splits = min(len(self.lengths), 3)
        split_method = KFold(n_splits)
        for components in range(self.min_n_components, self.max_n_components + 1):
            score = 0
            iter_counter = 0
            try:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    iter_counter += 1
                    X_train, len_train = combine_sequences(cv_train_idx, self.sequences)
                    X_test, len_test = combine_sequences(cv_test_idx, self.sequences)
                    hmm_model = GaussianHMM(n_components = components, covariance_type = 'diag', n_iter = 1000,
                                            random_state = self.random_state, verbose = False).fit(X_train, len_train)
                    score += self.model_score(hmm_model, X_test, len_test, components)
                    #print('Model fitted. score = ', score)
                score /= iter_counter if iter_counter > 0 else 1
                if score > best_score:
                    best_score = score
                    best_components_number = components
                    best_model = hmm_model
            except:
                pass
        if self.__class__.__name__ == 'SelectorCV':
            # need to train the model over all data available for Log likelihood model selection
            best_model = self.base_model(best_components_number)
        return best_model

    def model_score(model, X, length, n_components):
        """
        Returns different model scores for CV, BIC and DIC according to appropriate formulas. Returned score defined
        by class_name.
        Valid class names:
        'SelectorCV': Log likelihood
        'SelectorBIC': Bayesian information criteria
        'SelectorDIC': Discriminative information criteria
        """
        if self.__class__.__name__ == "SelectorCV":
            return model.score(X, length)
        if class_name == "SelectorBIC":
            #according to https://discussions.udacity.com/t/parameter-in-bic-selector/394318/2
            #BIC = -2 * LogL + p * LogN
            #LogN = log(len(X))
            #p = n_components^2 + 2*n_components*model.n_features - 1
            logL = model.score(X, length)
            logN = np.log(len(X))
            p = n_components**2 + 2 * n_components * model.n_features - 1
            return -2 * logL + p * logN
        return None        

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        return model_selector(self.__class__.__name__, self.lengths, self.min_n_components, self.max_n_components, 
                              self.sequences, self.random_state)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        #print(self.__class__.__name__)
        #print(super(self).__class__.__name__)
        return super(ModelSelector, self).select()
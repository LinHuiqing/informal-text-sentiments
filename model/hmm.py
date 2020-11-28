import numpy as np
import copy

STATES = [
    "B-postive", 
    "B-neutral", 
    "B-negative", 
    "I-postive", 
    "I-neutral", 
    "I-negative", 
    "O", 
]

class HMM():
    def __init__(self, possible_states=STATES):
        self.possible_states = possible_states

        self.count_emissions = {}
        self.count_transitions = {}
        self.count_states = {}
        self._init_state_dicts()

        self.vocab = {}

        self.emission_probabilities = None
        self.transition_probabilities = None

        self.k = 0.5

    def _init_state_dicts(self):
        for state in self.possible_states:
            self.count_emissions[state] = {}
            self.count_transitions[state] = {}
        return self
    
    def _count_all(self):
        for eg in self.train_data:
            store_state = None
            for x, y in eg:
                if self.count_emissions.get(x) == None:
                    self.count_emissions[x] = {}
                self.count_emissions[x][y] = self.count_emissions[x].get(y, 0) + 1
                # self.count_emissions[y][x] = self.count_emissions[y].get(x, 0) + 1
                if store_state != None:
                    self.count_transitions[store_state][y] = self.count_transitions[store_state].get(y) + 1

                # self.count_emissions[(y, x)] = self.count_emissions.get((y, x), 0) + 1
                self.count_states[y] = self.count_states.get(y, 0) + 1
                # if store_state != None:
                #     self.count_transitions[(store_state, x)] = self.count_transitions.get((store_state, x), 0) + 1
                self.vocab[x] = True
                store_state = x
        return self

    def _calculate_emission_prob(self, x, y):
        # self.emission_probabilities = copy.copy(self.count_emissions)
        # for y, y_dict in self.emission_probabilities.items():
        #     for x, count in y_dict.items():
        #         self.emission_probabilities[y][x] = count / self.count_states[y]
        return self.count_emissions[x][y] / self.count_states[y]
        # query = str(x)+" "+ str(y)
        # return emission_table[query]
    
    def _calculate_emission_prob_with_unk(self, x, y):
        # self.emission_probabilities = copy.copy(self.count_emissions)
        # for y, y_dict in self.emission_probabilities.items():
        #     for x, count in y_dict.items():
        #         if self.vocab.get(x, False):
        #             self.emission_probabilities[y][x] = count + self.k / self.count_states[y] + self.k
        #         else:
        #             self.emission_probabilities[y][x] = self.k / self.count_states[y] + self.k
        if self.count_emissions[y].get(x) != None:
            return self.count_emissions[x][y] + self.k / (self.count_states[y] + self.k)
        else:
            return self.k / (self.count_states[y] + self.k)

    def _get_argmax_y(self, probs_dict):
        return max(probs_dict, key=probs_dict.get)

    def _calculate_transition_prob(self, cur_y, next_y):
        # self.transition_probabilities = copy.copy(self.count_transitions)
        # for cur_y, cur_y_dict in self.transition_probabilities.items():
        #     for next_y, count in cur_y_dict.items():
        #         self.transition_probabilities[cur_y][next_y] = count / self.count_states[cur_y]
        # return self
        return self.count_transitions[cur_y][next_y] / self.count_states[cur_y]

    def train(self, train_data):
        self.train_data = train_data
        self._count_all()

    def predict(self, data):
        pass

    # def estimate_emission_with_unk(x,y):
    #     query = str(x)+" "+ str(y)
    #     if query not in emission_table.keys():
    #         # is an #UNK#
    #         print('is an #UNK#')
    #         return k/(tags[y]+k)
    #     else:
    #         return emission_table_k[query]
    

# O_tag_idx = 0 # position of O tag in emission list

# def calc_emission_probs():
#     """ To calculate the emission probabilities. (hq: hardcoded) 
    
#         Returns: 
#             Numpy array of emission probabilities. emission.shape = (# of examples x max # of words (assume 100) x # of types of emissions)
#     """
#     emissions = np.array(
#         [ 
#             [
#                 [1, 0, 0, 0, 0, 0, 0], 
#                 [0, 1, 0, 0, 0, 0, 0], 
#                 [0, 0, 1, 0, 0, 0, 0], 
#                 [0, 0, 0, 1, 0, 0, 0], 
#             ],
#             [
#                 [0, 0, 0, 0, 0, 1, 0], 
#                 [0, 1, 0, 0, 0, 0, 0], 
#                 [0, 0, 0, 0, 1, 0, 0], 
#                 [0, 0, 0, 1, 0, 0, 0], 
#             ],
#         ]
#         )
#     return emissions

# def predict_y(emissions):
#     """ To predict y values based on emission probabilities. 
        
#         Args:
#             emissions - probabilities of each emission per word per example.
#                 np.shape = (# of examples x max # of words (assume 100) x # of types of emissions)
        
#         Returns: 
#             Numpy array of index of maximum probability for each state of each example. 
#     """
#     labels = np.argmax(emissions, axis=2)
#     return labels

# def calc_precision_recall(y_true, y_pred):
#     """ To calculate precision and recall based on true and predicted y-values. 
    
#         Args: 
#             y_true - true labels of emissions.
#                 np.shape = (# of examples x max # of words (assume 100))
#             y_pred - predicted labels of emissions.
#                 np.shape = (# of examples x max # of words (assume 100))
        
#         Returns:
#             Precision and recall floats.
#     """
#     labelled_true = y_true != O_tag_idx
#     labelled_pred = y_pred != O_tag_idx

#     print(labelled_true)
#     print(labelled_pred)

#     correctly_pred = y_true == y_pred
#     correctly_pred = np.logical_and(correctly_pred, labelled_true)

#     no_of_correctly_pred = np.sum(correctly_pred)
#     precision = no_of_correctly_pred / np.sum(labelled_pred)
#     recall = no_of_correctly_pred / np.sum(labelled_true)

#     return precision, recall

# def calc_f_score(precision, recall):
#     """ To calculate the F score based on the precision and recall. 

#         Args: 
#             precision - precision score in float.
#             recall - recall score in float.
        
#         Returns: 
#             F score in float.
#     """
#     f_score = 2 / (1 / precision + 1 / recall)
#     return f_score

# if __name__ == "__main__":
#     emissions = calc_emission_probs()
#     print(emissions)

#     y_pred = predict_y(emissions)
#     print(y_pred)

#     y_true = np.array(
#         [
#             [2, 1, 1, 0],
#             [5, 1, 0, 1],
#         ]
#     )

#     precision, recall = calc_precision_recall(y_true, y_pred)
#     print("precision: {}".format(precision))
#     print("recall: {}".format(recall))

#     f_score = calc_f_score(precision, recall)
#     print("f_score: {}".format(f_score))

import numpy as np

O_tag_idx = 0 # position of O tag in emission list

def calc_emission_probs():
    """ To calculate the emission probabilities. (hq: hardcoded) 
    
        Returns: 
            Numpy array of emission probabilities. emission.shape = (# of examples x max # of words (assume 100) x # of types of emissions)
    """
    emissions = np.array(
        [ 
            [
                [1, 0, 0, 0, 0, 0, 0], 
                [0, 1, 0, 0, 0, 0, 0], 
                [0, 0, 1, 0, 0, 0, 0], 
                [0, 0, 0, 1, 0, 0, 0], 
            ],
            [
                [0, 0, 0, 0, 0, 1, 0], 
                [0, 1, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 1, 0, 0], 
                [0, 0, 0, 1, 0, 0, 0], 
            ],
        ]
        )
    return emissions

def predict_y(emissions):
    """ To predict y values based on emission probabilities. 
        
        Args:
            emissions - probabilities of each emission per word per example.
                np.shape = (# of examples x max # of words (assume 100) x # of types of emissions)
        
        Returns: 
            Numpy array of index of maximum probability for each state of each example. 
    """
    labels = np.argmax(emissions, axis=2)
    return labels

def calc_precision_recall(y_true, y_pred):
    """ To calculate precision and recall based on true and predicted y-values. 
    
        Args: 
            y_true - true labels of emissions.
                np.shape = (# of examples x max # of words (assume 100))
            y_pred - predicted labels of emissions.
                np.shape = (# of examples x max # of words (assume 100))
        
        Returns:
            Precision and recall floats.
    """
    labelled_true = y_true != O_tag_idx
    labelled_pred = y_pred != O_tag_idx

    print(labelled_true)
    print(labelled_pred)

    correctly_pred = y_true == y_pred
    correctly_pred = np.logical_and(correctly_pred, labelled_true)

    no_of_correctly_pred = np.sum(correctly_pred)
    precision = no_of_correctly_pred / np.sum(labelled_pred)
    recall = no_of_correctly_pred / np.sum(labelled_true)

    return precision, recall

def calc_f_score(precision, recall):
    """ To calculate the F score based on the precision and recall. 

        Args: 
            precision - precision score in float.
            recall - recall score in float.
        
        Returns: 
            F score in float.
    """
    f_score = 2 / (1 / precision + 1 / recall)
    return f_score

if __name__ == "__main__":
    emissions = calc_emission_probs()
    print(emissions)

    y_pred = predict_y(emissions)
    print(y_pred)

    y_true = np.array(
        [
            [2, 1, 1, 0],
            [5, 1, 0, 1],
        ]
    )

    precision, recall = calc_precision_recall(y_true, y_pred)
    print("precision: {}".format(precision))
    print("recall: {}".format(recall))

    f_score = calc_f_score(precision, recall)
    print("f_score: {}".format(f_score))

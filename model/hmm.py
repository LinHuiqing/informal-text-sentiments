import pprint

pp = pprint.PrettyPrinter()
class HMM():
    def __init__(self):
        """ Class implementing HMM model.

            Attributes:
                possible_states: List of possible states. E.g. 
                    [ "state_1", ..., "state_n" ]
                count_emissions: Dict containing emission counts. 
                                 E.g. 
                    {
                        "observation_1": {
                            "state_1": int, 
                            ...
                            "state_n": int, 
                        },
                        ...
                    }
                count_transitions: Dict containing transition counts. E.g.
                    {
                        "cureent_state_1": {
                            "next_state_1": int, 
                            ...
                            "next_state_n": int, 
                        },
                        ...
                    }
                count_states: Dict containing state counts. E.g.
                    {
                        "state_1": int, 
                        ...
                    }
                k: Int of constant k
                predictions: List of predictions made. E.g.
                    [
                        [
                            ["this", "state1"],
                            ["is", "state2"],
                            ["an", "state3"],
                            ["example", "state4"],
                        ],
                        ...
                    ]
        """
        self.possible_states = []

        self.count_emissions = {}
        self.count_transitions = {}
        self.count_states = {}

        self.k = 0.5

        self.predictions = []
    
    def _count_all(self):
        """ Count everything required, including: emissions, transitions and 
            states. 

            Returns: 
                self.
        """
        self.count_transitions["START"] = {}
        for eg in self.train_data:
            store_state = None
            for x, y in eg:
                if y not in self.possible_states:
                    self.possible_states.append(y)

                if self.count_emissions.get(x) == None:
                    self.count_emissions[x] = {}
                self.count_emissions[x][y] = self.count_emissions[x].get(y, 0) + 1

                if store_state != None:
                    if self.count_transitions.get(store_state) == None:
                        self.count_transitions[store_state] = {}
                    self.count_transitions[store_state][y] = self.count_transitions[store_state].get(y, 0) + 1
                else:
                    self.count_transitions["START"][y] = self.count_transitions["START"].get(y, 0) + 1
                self.count_states[y] = self.count_states.get(y, 0) + 1
                store_state = y
            self.count_transitions[store_state]["STOP"] = self.count_transitions[store_state].get("STOP", 0) + 1
        return self

    def _calculate_emission_prob(self, x, y):
        """ Calculate emission probabilities.

            Args:
                x: observation.
                y: state.

            Returns:
                Float of emission probability.
        """
        return self.count_emissions.get(x, {}).get(y, 0) / self.count_states[y]
    
    def _calculate_emission_prob_with_unk(self, x, y):
        """ Calculate emission probabilities with unknown tokens.

            Args:
                x: observation.
                y: state.
            
            Returns:
                Float of emission probability.
        """
        if self.count_emissions.get(x) != None:
            return self.count_emissions.get(x, {}).get(y, 0) / (self.count_states[y] + self.k)
        else:
            return self.k / (self.count_states.get(y, 0) + self.k)

    def _get_argmax_y(self, probs_dict):
        """ Get argmax y; y with highest probability.

            Args:
                probs_dict: dict of all probabilities given states.

            Returns:
                Str of y with the highest probability.
        """
        return max(probs_dict, key=probs_dict.get)

    def _calculate_transition_prob(self, cur_y, next_y):
        """ Calculate the transition probabilities.

            Args:
                cur_y: str of current state.
                next_y: str of next state.
            
            Returns:
                Float of transition probability to move from cur_y to next_y.
        """
        return self.count_transitions[cur_y][next_y] / self.count_states[cur_y]

    def train(self, train_data):
        """ Train model.

            Args: 
                train_data: list of parsed training data of the following form:
                    [
                        [
                            ["this", "state1"],
                            ["is", "state2"],
                            ["an", "state3"],
                            ["example", "state4"],
                        ],
                        ...
                    ]
            
            Returns:
                self.
        """
        self.train_data = train_data
        self._count_all()
        return self

    def predict_part2(self, data):
        """ Predict states for part 2. 

            Args:
                data: list of parsed dev data of the following form:
                    [
                        [ "This", "is", "an", "example" ],
                        ...
                    ]
            
            Returns:
                self.
        """
        self.predictions = []
        for eg in data:
            eg_pred = []
            for observation in eg:
                probabilities = {}
                for state in self.possible_states:
                    probabilities[state] = self._calculate_emission_prob_with_unk(observation, state)
                state = self._get_argmax_y(probabilities)
                eg_pred.append([observation, state])
            self.predictions.append(eg_pred)
        return self
    
    def write_preds(self, filename):
        """ Write predictions to file.

            Args:
                filename: filename to write predictions to.
            
            Returns:
                self.
        """
        output = ""
        for eg in self.predictions:
            for entity in eg:
                output += "{}\n".format(" ".join(entity))
            output += "\n"
        output += "\n"

        with open(filename, "w") as f:
            f.write(output)
        return self
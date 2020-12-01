class HMM():
    def __init__(self):
        self.possible_states = []

        self.count_emissions = {}
        self.count_transitions = {}
        self.count_states = {}

        self.k = 0.5

        self.pred = []
    
    def _count_all(self):
        """ Count everything required.

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
        return self.count_emissions.get(x, {}).get(y, 0) / self.count_states[y]
    
    def _calculate_emission_prob_with_unk(self, x, y):
        if self.count_emissions.get(x, {}).get(y) != None:
            return self.count_emissions[x][y] + self.k / (self.count_states[y] + self.k)
        else:
            return self.k / (self.count_states.get(y, 0) + self.k)

    def _get_argmax_y(self, probs_dict):
        return max(probs_dict, key=probs_dict.get)

    def _calculate_transition_prob(self, cur_y, next_y):
        return self.count_transitions[cur_y][next_y] / self.count_states[cur_y]

    def train(self, train_data):
        self.train_data = train_data
        self._count_all()

    def predict_part2(self, data):
        self.pred = []
        for eg in data:
            eg_pred = []
            for observation in eg:
                probabilities = {}
                for state in self.possible_states:
                    probabilities[state] = self._calculate_emission_prob(observation, state)
                state = self._get_argmax_y(probabilities)
                eg_pred.append([observation, state])
            self.pred.append(eg_pred)
    
    def write_preds(self, filename):
        output = ""
        for eg in self.pred:
            for entity in eg:
                output += "{}\n".format(" ".join(entity))
            output += "\n"
        output += "\n"

        with open(filename, "w") as f:
            f.write(output)

import math
class HMM:
    def __init__(self, k=0.5):
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

        self.k = k

        self.predictions = None

        self.emission_probabilities = {}
        self.transition_probabilities = {}
    
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
                    self.count_states["START"] = self.count_states.get("START", 0) + 1
                self.count_states[y] = self.count_states.get(y, 0) + 1
                store_state = y
            self.count_transitions[store_state]["STOP"] = self.count_transitions[store_state].get("STOP", 0) + 1
        return self

    def _count_all_good_turing(self):
        """ Count everything required, including: emissions, transitions and 
            states. 

            Returns: 
                self.
        """
        self._count_all()
        self.emission_count_freq = {}
        self.count_observations = 0
        for _, state_dict in self.count_emissions.items():
            for _, count in state_dict.items():
                self.emission_count_freq[count] = self.emission_count_freq.get(count, 0) + 1
                self.count_observations += count
        return self

    def _calculate_emission_prob(self, x, y):
        """ Calculate emission probabilities (without unknown tokens).

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
        if self.count_emissions.get(x) == None:
            x = "#UNK#"
        
        self.emission_probabilities[x] = self.emission_probabilities.get(x, {})
        if self.emission_probabilities[x].get(y) != None:
            return self.emission_probabilities[x][y]
        
        if x == "#UNK#":
            self.emission_probabilities[x][y] = self.k / (self.count_states.get(y, 0) + self.k)
        else:
            self.emission_probabilities[x][y] = self.count_emissions.get(x, {}).get(y, 0) / (self.count_states[y] + self.k)

        return self.emission_probabilities[x][y]

    def _calculate_emission_prob_with_unk_laplace_smoothing(self, x, y):
        """ Calculate emission probabilities with laplace smoothing.

            Args:
                x: observation.
                y: state.
            
            Returns:
                Float of emission probability.
        """
        if self.count_emissions.get(x) == None:
            x = "#UNK#"
        
        self.emission_probabilities[x] = self.emission_probabilities.get(x, {})
        if self.emission_probabilities[x].get(y) != None:
            return self.emission_probabilities[x][y]
        
        self.emission_probabilities[x][y] = (self.count_emissions.get(x, {}).get(y, 0) + 1) / (self.count_states[y] + len(self.count_emissions.keys()) + 1)

        return self.emission_probabilities[x][y]

    def _calculate_emission_prob_with_unk_good_turing(self, x, y):
        """ Calculate emission probabilities with good turing smoothing.

            Args:
                x: observation.
                y: state.
            
            Returns:
                Float of emission probability.
        """
        if self.count_emissions.get(x) == None:
            x = "#UNK#"
        
        self.emission_probabilities[x] = self.emission_probabilities.get(x, {})
        if self.emission_probabilities[x].get(y) != None:
            return self.emission_probabilities[x][y]
        
        if x == "#UNK#" or self.count_emissions[x].get(y) == None:
            self.emission_probabilities[x][y] = self.emission_count_freq[1] / self.count_observations
        else:
            # self.emission_probabilities[x][y] = self.count_emissions.get(x, {}).get(y, 0) / (self.count_states[y] + self.k)
            c = self.count_emissions[x][y]
            self.emission_probabilities[x][y] = ((c + 1) * self.emission_count_freq.get(c+1, 1)) / self.emission_count_freq[c]

        return self.emission_probabilities[x][y]

    def _get_argmax_y_part2(self, x):
        """ Get argmax y; y with highest probability. (Part 2: only emission probabilities)

            Args:
                probs_dict: dict of all probabilities given states.

            Returns:
                Str of y with the highest probability.
        """
        probabilities = {}
        for state in self.possible_states:
            probabilities[state] = self._calculate_emission_prob_with_unk(x, state)
        return max(probabilities, key=probabilities.get)
    

    def _calculate_transition_prob(self, cur_y, next_y):
        """ Calculate the transition probabilities.

            Args:
                cur_y: str of current state.
                next_y: str of next state.
            
            Returns:
                Float of transition probability to move from cur_y to next_y.
        """
        self.transition_probabilities[cur_y] = self.transition_probabilities.get(cur_y, {})
        if self.transition_probabilities[cur_y].get(next_y) == None:
            self.transition_probabilities[cur_y][next_y] = self.count_transitions[cur_y].get(next_y, 0) / self.count_states[cur_y]
        return self.transition_probabilities[cur_y][next_y]

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
    
    def train_good_turing(self, train_data):
        """ Train model with good turing.

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
        self._count_all_good_turing()
        max_freq = max(self.emission_count_freq.keys())
        for freq in range(1, max_freq):
            if self.emission_count_freq.get(freq) == None:
                self.emission_count_freq[freq] = self.emission_count_freq[freq-1]
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
                state = self._get_argmax_y_part2(observation)
                eg_pred.append([observation, state])
            self.predictions.append(eg_pred)
        return self

    def _viterbi_algo(self, example):
        """ Implementation of viterbi algorithm.

            Args:
                example: List of strings for words in example.
            
            Returns:
                List of predictions for example.
        """
        # Base Case
        pi = {
            0: {
                "START": 0,
            }
        }
        k = 1

        # Move forward recursively
        for observation in example:
            pi[k] = {}
            for v in self.possible_states:
                probabilities = {}
                for u in pi[k-1].keys():
                    trans_prob = self._calculate_transition_prob(u, v)
                    em_prob = self._calculate_emission_prob_with_unk(observation, v)
                    if trans_prob > 0 and em_prob > 0:
                        probabilities[u] = pi[k-1][u] \
                            + math.log(trans_prob) \
                            + math.log(em_prob)
                    else:
                        probabilities[u] = float("-inf")
                    
                max_val = max(probabilities.values())
                pi[k][v] = max_val
            k += 1
        
        # Transition to STOP
        probabilities = {}
        for u in pi[k-1].keys():
            trans_prob = self._calculate_transition_prob(u, "STOP")
            if trans_prob > 0:
                probabilities[u] = pi[k-1][u] \
                    + math.log(trans_prob)
        
        # Best y_n
        y_n = max(probabilities, key=probabilities.get)
        state_pred_r = [y_n]

        # Backtrack
        for n in reversed(range(1, k)):
            probabilities = {}
            for v in pi[n-1].keys():
                trans_prob = self._calculate_transition_prob(v, state_pred_r[-1])
                if trans_prob > 0:
                    probabilities[v] = pi[n-1][v] + math.log(trans_prob)
            if all(prob == float("-inf") for prob in probabilities.values()):
                state_pred_r.append("O")
            else:
                state_pred_r.append(max(probabilities, key=probabilities.get))

        # Prepare output
        prediction = []
        for idx, observation in enumerate(example):
            prediction.append([observation, state_pred_r[k-idx-2]])
        
        return prediction
    
    def _viterbi_algo_3rdbest(self, example):
        """ Implementation of viterbi algorithm for Part 4 (get 3rd best output sequence).

            Args:
                example: List of strings for words in example.
            
            Returns:
                List of predictions for example.
        """
        # Base Case
        pi = {
            0: {
                "START": [[1,[]]],
            }
        }

        k = 1

        # Move forward recursively
        for observation in example:
            pi[k] = {}
            for v in self.possible_states:
                probabilities = {}
                for u in pi[k-1].keys():
                    trans_prob = self._calculate_transition_prob(u, v)
                    em_prob = self._calculate_emission_prob_with_unk(observation, v)
                    if trans_prob > 0 and em_prob > 0:
                        for i in range(len(pi[k-1][u])):
                            probabilities[tuple(pi[k-1][u][i][1]+[u])] = pi[k-1][u][i][0] \
                            + math.log(trans_prob) \
                            + math.log(em_prob)

                    else:
                        # probabilities[u] = float("-inf")
                        for i in range(len(pi[k-1][u])):
                            probabilities[tuple(pi[k-1][u][i][1]+[u])] = float("-inf")
        
                
                probabilitiesValues = sorted(probabilities.values(), reverse=True)
                probabilitiesKeys = sorted(probabilities, key=probabilities.get, reverse=True)
                if len(probabilitiesValues) == 1:
                    pi[k][v] = [[probabilitiesValues[0], list(probabilitiesKeys[0])]]
                else:
                    pi[k][v] = [[probabilitiesValues[0], list(probabilitiesKeys[0])], [probabilitiesValues[1], list(probabilitiesKeys[1])], [probabilitiesValues[2], list(probabilitiesKeys[2])]]

            k += 1
        
        # Transition to STOP
        probabilities = {}
        for u in pi[k-1].keys():
            trans_prob = self._calculate_transition_prob(u, "STOP")
            if trans_prob > 0:
                for i in range(len(pi[k-1][u])):
                            probabilities[tuple(pi[k-1][u][i][1]+[u])] = pi[k-1][u][i][0] \
                            + math.log(trans_prob)
            else:
                for i in range(len(pi[k-1][u])):
                            probabilities[tuple(pi[k-1][u][i][1]+[u])] = float("-inf")

        probabilitiesValues = sorted(probabilities.values(), reverse=True)
        probabilitiesKeys = sorted(probabilities, key=probabilities.get, reverse=True)
        thirdBestPath = probabilitiesKeys[2]

        # Prepare output
        prediction = []
        for idx, observation in enumerate(example):
            prediction.append([observation, thirdBestPath[idx+1]])
        return prediction

    def _viterbi_algo_laplace_smoothing(self, example):
        """ Implementation of viterbi algorithm with laplace smoothing.

            Args:
                example: List of strings for words in example.
            
            Returns:
                List of predictions for example.
        """
        # Base Case
        pi = {
            0: {
                "START": 0,
            }
        }
        k = 1

        # Move forward recursively
        for observation in example:
            pi[k] = {}
            for v in self.possible_states:
                probabilities = {}
                for u in pi[k-1].keys():
                    trans_prob = self._calculate_transition_prob(u, v)
                    em_prob = self._calculate_emission_prob_with_unk_laplace_smoothing(observation, v)
                    if trans_prob > 0 and em_prob > 0:
                        probabilities[u] = pi[k-1][u] \
                            + math.log(trans_prob) \
                            + math.log(em_prob)
                    else:
                        probabilities[u] = float("-inf")
                max_val = max(probabilities.values())
                pi[k][v] = max_val
            k += 1
        
        # Transition to STOP
        probabilities = {}
        for u in pi[k-1].keys():
            trans_prob = self._calculate_transition_prob(u, "STOP")
            if trans_prob > 0:
                probabilities[u] = pi[k-1][u] \
                    + math.log(trans_prob)
        
        # Best y_n
        y_n = max(probabilities, key=probabilities.get)
        state_pred_r = [y_n]

        # Backtrack
        for n in reversed(range(1, k)):
            probabilities = {}
            for v in pi[n-1].keys():
                trans_prob = self._calculate_transition_prob(v, state_pred_r[-1])
                if trans_prob > 0:
                    probabilities[v] = pi[n-1][v] + math.log(trans_prob)
            state_pred_r.append(max(probabilities, key=probabilities.get))

        # Prepare output
        prediction = []
        for idx, observation in enumerate(example):
            prediction.append([observation, state_pred_r[k-idx-2]])
        
        return prediction
    
    def _viterbi_algo_good_turing(self, example):
        """ Implementation of viterbi algorithm with good turing.

            Args:
                example: List of strings for words in example.
            
            Returns:
                List of predictions for example.
        """
        # Base Case
        pi = {
            0: {
                "START": 0,
            }
        }
        k = 1

        # Move forward recursively
        for observation in example:
            pi[k] = {}
            for v in self.possible_states:
                probabilities = {}
                for u in pi[k-1].keys():
                    trans_prob = self._calculate_transition_prob(u, v)
                    em_prob = self._calculate_emission_prob_with_unk_good_turing(observation, v)
                    if trans_prob > 0 and em_prob > 0:
                        probabilities[u] = pi[k-1][u] \
                            + math.log(trans_prob) \
                            + math.log(em_prob)
                    else:
                        probabilities[u] = float("-inf")
                max_val = max(probabilities.values())
                pi[k][v] = max_val
            k += 1
        
        # Transition to STOP
        probabilities = {}
        for u in pi[k-1].keys():
            trans_prob = self._calculate_transition_prob(u, "STOP")
            if trans_prob > 0:
                probabilities[u] = pi[k-1][u] \
                    + math.log(trans_prob)
        
        # Best y_n
        y_n = max(probabilities, key=probabilities.get)
        state_pred_r = [y_n]

        # Backtrack
        for n in reversed(range(1, k)):
            probabilities = {}
            for v in pi[n-1].keys():
                trans_prob = self._calculate_transition_prob(v, state_pred_r[-1])
                if trans_prob > 0:
                    probabilities[v] = pi[n-1][v] + math.log(trans_prob)
            state_pred_r.append(max(probabilities, key=probabilities.get))

        # Prepare output
        prediction = []
        for idx, observation in enumerate(example):
            prediction.append([observation, state_pred_r[k-idx-2]])
        
        return prediction

    def predict_part3(self, data):
        """ Predict states for part 3. 

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
            eg_pred = self._viterbi_algo(eg)
            self.predictions.append(eg_pred)
        return self

    def predict_part4(self, data):
        """ Predict states for part 4. 

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
            eg_pred = self._viterbi_algo_3rdbest(eg)
            self.predictions.append(eg_pred)
        return self
    
    def predict_part5_laplace_smoothing(self, data):
        """ Predict states for part 3 with laplace smoothing. 

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
            eg_pred = self._viterbi_algo_laplace_smoothing(eg)
            self.predictions.append(eg_pred)
        return self
    
    def predict_part5_good_turing(self, data):
        """ Predict states for part 3 with good turing. 

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
            eg_pred = self._viterbi_algo_good_turing(eg)
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
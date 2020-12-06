from model.hmm import HMM

class Trigram(HMM):
    def run_trigram(self,example):
                # Base Case
        pi = {
            -1: {
                "START": 1,
            },
            0: {
                "START": 1,
            }
        }
        k = 1

        # Assume the trigram is u -> v -> w
        # Move forward recursively
        for observation in example:
            pi[k] = {}
            for w in self.possible_states:
                probabilities = {}
                for u in pi[k-1].keys():
                    for v in self.possible_states:
                        probabilities_v = {}
                        trans_prob_uv = self._calculate_transition_prob(u, v)
                        trans_prob_uvw = self._calculate_transition_prob(v, w)
                        trans_prob = trans_prob_uv * trans_prob_uvw
                        em_prob = self._calculate_emission_prob_with_unk(observation, w)
                        if trans_prob_uvw > 0 and em_prob > 0:
                            probabilities_v[w] = pi[k-1][u] \
                                + math.log(trans_prob) \
                                + math.log(em_prob)
                        else:
                            probabilities_v[w] = float("-inf")
                    probabilities[w] = max(probabilities_v.values())
                    max_val = max(probabilities.values())
                pi[k][u] = max_val
            k += 1
        
        # Transition to STOP
        probabilities = {}
        for u in pi[k-1].keys():
            for v in self.possible_states:
                probabilities_v = {}
                trans_prob_uv = self._calculate_transition_prob(u, v)
                trans_prob_uvSTOP = self._calculate_transition_prob(v, "STOP")
                trans_prob = trans_prob_uv * trans_prob_uvSTOP
                if (trans_prob) > 0:
                    probabilities_v[v] = pi[k-1][u] \
                        + math.log(trans_prob)
            probabilities[u] = max(probabilities_v.values())
        
        # Best y_n
        y_n = max(probabilities, key=probabilities.get)
        state_pred_r = [y_n]

        # Backtrack
        for n in reversed(range(1, k)):
            probabilities = {}
            for v in pi[n-1].keys():
                probabilities_w = {}
                for w in pi[n-2].keys():
                    trans_prob_wv = self._calculate_transition_prob(w, v)
                    trans_prob_vend = self._calculate_transition_prob(v, state_pred_r[-1])
                    trans_prob = trans_prob_wv * trans_prob_vend
                if trans_prob > 0:
                    probabilities_w[w] = pi[n-1][v] + math.log(trans_prob)
                probablities[v] = max(probabilities_w)
            state_pred_r.append(max(probabilities, key=probabilities.get))

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


import numpy as np
class HMM:
    def __init__(self, A, B, Pi):
        self.A = A
        self.B = B
        self.Pi = Pi

    def broadcast(self, dp, res, T, O):
        for t in range(1, T):
            dp[t] = np.sum(dp[t - 1] * self.A, axis=1) * self.B[:, O[t]]
            res += np.log(1 / np.sum(dp[t]))
            dp[t] /= np.sum(dp[t])
        return res

    def nested(self, dp, res, T, O, N):
        for t in range(1, T):
            for i in range(N):
                dp[t, i] = np.sum(dp[t - 1] * self.A[:, i]) * self.B[i, O[t]]
            res += np.log(1 / np.sum(dp[t]))
            dp[t] /= np.sum(dp[t])
        return res

    def forward_log(self, O: list):
        """
        :param O: is the sequence (an array of) discrete (integer) observations, i.e. [0, 2,1 ,3, 4]
        :return: ln P(O|λ) score for the given observation, ln: natural logarithm
        """
        T = len(O)
        N = self.A.shape[0]
        dp = np.zeros((T, N))
        dp[0] = self.Pi * self.B[:, O[0]]
        res = np.log(1/np.sum(dp[0]))
        dp[0] /= np.sum(dp[0])
        # For broadcasting case please open following line but not give the exact results (Gives close results)
        # res = self.broadcast(dp, res, T, O)
        # For nested case please open following line
        res = self.nested(dp, res, T, O, N)
        return -res

    def dict_maker(self, T, dict_list, N, O):
        for t in range(1, T):
            dict_list.append({})
            for s in range(N):
                max_prob = float('-inf')
                max_prev = None
                for prev_s in range(N):
                    prob = dict_list[t - 1][prev_s]["p"] + np.log(self.A[prev_s, s]) + np.log(self.B[s, O[t]])
                    if prob > max_prob:
                        max_prob = prob
                        max_prev = prev_s
                dict_list[t][s] = {"p": max_prob, "pr": max_prev}
        return max(prob["p"] for prob in dict_list[-1].values()), dict_list

    def viterbi_log(self, O: list):
        """
        :param O: is an array of discrete (integer) observations, i.e. [0, 2,1 ,3, 4]
        :return: the tuple (Q*, ln P(Q*|O,λ)), Q* is the most probable state sequence for the given O
        """
        T = len(O)
        N = self.A.shape[0]
        dict_list = [{i: {"p": np.log(self.Pi[i]) + np.log(self.B[i, O[0]]), "pr": None} for i in range(N)}]
        max_prob, dict_list = self.dict_maker(T, dict_list, N, O)
        prev = None
        max_list = []
        for item, info in dict_list[-1].items():
            if info["p"] == max_prob:
                max_list.append(item)
                prev = item
        i = len(dict_list) - 2
        while i >= 0:
            max_list.insert(0, dict_list[i + 1][prev]["pr"])
            prev = dict_list[i + 1][prev]["pr"]
            i -= 1
        return max_prob, max_list

import numpy as np
import pickle
import matplotlib.pyplot as plt

display = False

def Borda_Count(preference):
    voter_num, candidate_num = preference.shape
    score = np.zeros(candidate_num, dtype=int)
    for i in range(voter_num):
        winners = sorted(range(candidate_num), key=lambda k: -preference[i][k])
        for j in range(candidate_num):
            score[winners[j]] += (candidate_num - j)
    
    if display:
        print(score)

    winners = sorted(range(candidate_num), key=lambda k: -score[k])
    if score[winners[0]] == score[winners[1]]:
        return None # tie
    return winners[0]

def Condorcet_Voting(preference):
    def Condorcet_winner(c):
        for k in range(candidate_num):
            if k != c and preferred_over[k][c]:
                return False
        return True
    
    voter_num, candidate_num = preference.shape

    preferred_over = np.zeros((candidate_num, candidate_num), dtype=bool)
    for i in range(candidate_num):
        for j in range(i + 1, candidate_num):
            count = 0
            for voter in range(voter_num):
                if preference[voter][i] > preference[voter][j]:
                    count += 1
            if count < (voter_num / 2):
                # j preferred over i
                preferred_over[j][i] = True
            elif count > (voter_num / 2):
                preferred_over[i][j] = True

        winner = None
        for c in range(candidate_num):
            if winner == None and Condorcet_winner(c):
                winner = c

    return winner

def Majority_Judgment(preference):
    voter_num, candidate_num = preference.shape
    medians = np.zeros(candidate_num)
    for j in range(candidate_num):
        medians[j] = np.median(preference[:, j])
    
    if display:
        print(medians)

    winners = sorted(range(candidate_num), key=lambda k: -medians[k])
    if medians[winners[0]] == medians[winners[1]]:
        return None # tie
    return winners[0]

def Single_Transferable_Vote(preference):
    voter_num, candidate_num = preference.shape
    
    remaining_candidates = set(range(candidate_num))
    for round in range(candidate_num - 1):
        votes = np.zeros(candidate_num, dtype=int)
        for i in range(voter_num):
            max_preference, max_candidate = 0.0, None
            for j in remaining_candidates:
                if preference[i][j] > max_preference:
                    max_preference = preference[i][j]
                    max_candidate = j
            votes[max_candidate] += 1

        eliminate = None
        for c in remaining_candidates:
            if eliminate == None or votes[c] < votes[eliminate]:
                eliminate = c

        if display:
            print(eliminate)

        remaining_candidates.discard(eliminate)

    return remaining_candidates.pop()

def k_Approval_Voting(preference, k):
    voter_num, candidate_num = preference.shape
    votes = np.zeros(candidate_num, dtype=int)
    for i in range(voter_num):
        winners = sorted(range(candidate_num), key=lambda k: -preference[i][k]) # sort in descending order
        for j in range(k):
            votes[winners[j]] += 1
    
    winners = sorted(range(candidate_num), key=lambda k: -votes[k])
    if votes[winners[0]] == votes[winners[1]]:
        return None # tie
    return winners[0]

def generate_preference(voter_num, candidate_num):
    preference = np.zeros((voter_num, candidate_num))
    voters = np.random.rand(voter_num) # uniform distribution over [0, 1)
    candidates = np.random.rand(candidate_num)

    for i in range(voter_num):
        preference[i] = score(voters[i], candidates[j])

    return preference

found = False
candidate_num = 5
N = 4

while not found:
    preference = np.zeros((100, candidate_num))
    preference_list = []
    for i in range(N):
        preference_list.append(np.random.permutation(candidate_num))
    count = np.zeros(N, dtype=int)
    for i in range(100):
        RI = np.random.randint(N)
        count[RI] += 1
        preference[i] = preference_list[RI]

    W1 = [k_Approval_Voting(preference, 1), k_Approval_Voting(preference, 2), k_Approval_Voting(preference, 3), k_Approval_Voting(preference, 4)]

    B = Borda_Count(preference)
    C = Condorcet_Voting(preference)
    M = Majority_Judgment(preference)
    S = Single_Transferable_Vote(preference)

    W2 = [B, C, M, S]

    print(W1)
    print(W2)

    if None not in W1 and None not in W2:
        W1.sort()
        W2.sort()

        found = True
        for i in range(4):
            if W1[i] != W2[i]:
                found = False
            if i > 0 and W1[i] == W1[i - 1]:
                found = False
        
        if found:
            print(preference_list)
            print(count)

            display = True

            B = Borda_Count(preference)
            print("Borda Count :", B)
            C = Condorcet_Voting(preference)
            print("Condorcet Voting :", C)
            M = Majority_Judgment(preference)
            print("Majority Judgment :", M)
            S = Single_Transferable_Vote(preference)
            print("Single Transferable Vote :", S)

'''
[3, 1, 4, 0]
[3, 4, 1, 0]
[array([1, 3, 2, 0, 4]), array([4, 3, 0, 1, 2]), array([1, 0, 3, 4, 2]), array([2, 1, 3, 4, 0])]
[24 30 28 18]
[308 280 286 314 312]
Borda Count : 3
Condorcet Voting : 4
[1. 3. 2. 1. 2.]
Majority Judgment : 1
1
2
4
3
Single Transferable Vote : 0
'''
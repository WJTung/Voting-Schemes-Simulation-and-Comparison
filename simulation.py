import numpy as np
import pickle
import matplotlib.pyplot as plt

load = True
display = False

noise_intensity = 0.05

min_candidate_num = 3
max_candidate_num = 10

election_num = 1000

scheme_num = 4
schemes = ["Borda Count", "Condorcet Voting", "Majority Judgment", "Single Transferable Vote"]

plt.rcParams['font.sans-serif'] = ['Roboto']
plt.rcParams['font.size'] = 12

def score(x1, x2):
    distance = abs(x1 - x2)
    noise = np.random.normal() * noise_intensity
    if distance > 0.4:
        return 1.0 - (10.0 / 7.0) * 0.4 - (5.0 / 7.0) * (distance - 0.4) + noise
    else:
        return 1.0 - (10.0 / 7.0) * distance + noise

def generate_preference(voter_num, candidate_num):
    preference = np.zeros((voter_num, candidate_num))
    voters = np.random.rand(voter_num) # uniform distribution over [0, 1)
    candidates = np.random.rand(candidate_num)

    for i in range(voter_num):
        for j in range(candidate_num):
            preference[i, j] = score(voters[i], candidates[j])

    return preference

# Borda_Count, Condorcet_Voting, Majority_Judgment, Single Transferable Vote : return rank

def Borda_Count(preference, candidate):
    voter_num, candidate_num = preference.shape
    score = np.zeros(candidate_num, dtype=int)
    for i in range(voter_num):
        winners = sorted(range(candidate_num), key=lambda k: -preference[i][k])
        for j in range(candidate_num):
            score[winners[j]] += (candidate_num - j)
    
    winners = sorted(range(candidate_num), key=lambda k: -score[k])
    for rank in range(candidate_num):
        if winners[rank] == candidate:
            return rank

def Condorcet_Voting(preference, candidate):
    def Condorcet_winner(c):
        for k in remaining_candidates:
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

    remaining_candidates = set(range(candidate_num))
    for rank in range(candidate_num):
        winner = None
        for c in remaining_candidates:
            if winner == None and Condorcet_winner(c):
                winner = c
        if winner == None:
            return None
        remaining_candidates.discard(winner)
        if winner == candidate:
            return rank
    
    return None

def Majority_Judgment(preference, candidate):
    voter_num, candidate_num = preference.shape
    medians = np.zeros(candidate_num)
    for j in range(candidate_num):
        medians[j] = np.median(preference[:, j])
    
    winners = sorted(range(candidate_num), key=lambda k: -medians[k])
    for rank in range(candidate_num):
        if winners[rank] == candidate:
            return rank

def Single_Transferable_Vote(preference, candidate):
    voter_num, candidate_num = preference.shape
    
    remaining_candidates = set(range(candidate_num))
    for round in range(candidate_num):
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

        remaining_candidates.discard(eliminate)
        if eliminate == candidate:
            rank = candidate_num - round - 1
            return rank

def k_Approval_Voting(preference, k):
    voter_num, candidate_num = preference.shape
    votes = np.zeros(candidate_num, dtype=int)
    for i in range(voter_num):
        winners = sorted(range(candidate_num), key=lambda k: -preference[i][k]) # sort in descending order
        for j in range(k):
            votes[winners[j]] += 1
    
    winner = np.argmax(votes)
    return winner

# main

if not load:
    rank = np.zeros((max_candidate_num + 1, max_candidate_num + 1, 4, max_candidate_num), dtype=int) # candidate_num, k, 4 other voting schemes, rank

    candidate_num = min_candidate_num

    while candidate_num <= max_candidate_num:
        print("candidate_num =", candidate_num)

        k = 1
        while k < candidate_num:
            print("k =", k)

            count = 0
            for t in range(election_num):
                preference = generate_preference(voter_num=256, candidate_num=candidate_num)

                k_Approval_Voting_winner = k_Approval_Voting(preference, k=k)

                Borda_Count_rank = Borda_Count(preference, k_Approval_Voting_winner)
                Condorcet_Voting_rank = Condorcet_Voting(preference, k_Approval_Voting_winner)
                Majority_Judgment_rank = Majority_Judgment(preference, k_Approval_Voting_winner)
                Single_Transferable_Vote_rank = Single_Transferable_Vote(preference, k_Approval_Voting_winner)

                rank[candidate_num, k, 0, Borda_Count_rank] += 1
                if Condorcet_Voting_rank != None:
                    rank[candidate_num, k, 1, Condorcet_Voting_rank] += 1
                else:
                    count += 1
                rank[candidate_num, k, 2, Majority_Judgment_rank] += 1
                rank[candidate_num, k, 3, Single_Transferable_Vote_rank] += 1

            k += 1

            print(count, " elections no Condorcet rank")

        candidate_num += 1

    pickle.dump(rank, open("rank", "wb"))

rank = pickle.load(open("rank", "rb"))

max1, max2 = [[] for i in range(4)], [[] for i in range(4)]

# efficiency
candidate_num = min_candidate_num
while candidate_num <= max_candidate_num:
    title = str(candidate_num) + " candidates efficiency"
    plt.title(title, fontsize=20, weight="bold")
    ax = plt.gca()
    x = list(range(1, candidate_num))
    for scheme in range(scheme_num):
        y = []
        for k in x:
            y.append(rank[candidate_num, k, scheme, 0] * 100.0 / np.sum(rank[candidate_num, k, scheme]))
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(x, y, marker='.', markersize=12, color=color)
        max_index = np.argmax(y)
        max1[scheme].append(x[max_index])
        plt.plot(x[max_index], y[max_index], marker='*', markersize=16, color=color, zorder=3, label="_nolegend_")
    plt.xlabel("k", fontsize=14, weight="bold")
    plt.xticks(x)
    plt.ylabel("percentage (%)", fontsize=14, weight="bold")
    plt.ylim([0, 100])
    plt.legend(schemes, markerscale=0)
    if display:
        plt.show()
    else:
        fig_name = title + ".png"
        plt.savefig(fig_name)
        plt.cla()
    
    candidate_num += 1

# average
candidate_num = min_candidate_num
while candidate_num <= max_candidate_num:
    title = str(candidate_num) + " candidates average"
    plt.title(title, fontsize=20, weight="bold")
    ax = plt.gca()
    x = list(range(1, candidate_num))
    for scheme in range(scheme_num):
        y = []
        for k in x:
            num, sum = 0, 0
            for r in range(candidate_num):
                num += rank[candidate_num, k, scheme, r]
                sum += rank[candidate_num, k, scheme, r] * (candidate_num - r)
            y.append(sum / num)
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(x, y, marker='.', markersize=12, color=color)
        max_index = np.argmax(y)
        max2[scheme].append(x[max_index])
        plt.plot(x[max_index], y[max_index], marker='D', markersize=10, color=color, zorder=3, label="_nolegend_")
    plt.xlabel("k", fontsize=14, weight="bold")
    plt.xticks(x)
    plt.ylabel("average score", fontsize=14, weight="bold")
    plt.ylim([0, candidate_num])
    plt.legend(schemes, markerscale=0)
    if display:
        plt.show()
    else:
        fig_name = title + ".png"
        plt.savefig(fig_name)
        plt.cla()

    candidate_num += 1

# schemes
for scheme in range(scheme_num):
    title = schemes[scheme]
    plt.title(title, fontsize=20, weight="bold")
    ax = plt.gca()
    candidate_num = min_candidate_num
    x = []
    while candidate_num <= max_candidate_num:
        x.append(candidate_num)
        candidate_num += 1

    plt.plot(x, max2[scheme], marker='D', markersize=10)
    plt.plot(x, max1[scheme], marker='*', markersize=12)
    plt.xlabel("number of candidates", fontsize=14, weight="bold")
    plt.xticks(x)
    plt.yticks(list(range(0, max_candidate_num + 1)))
    plt.ylabel("best k", fontsize=14, weight="bold")
    plt.legend(["efficiency", "average"])
    if display:
        plt.show()
    else:
        fig_name = title + ".png"
        plt.savefig(fig_name)
        plt.cla()
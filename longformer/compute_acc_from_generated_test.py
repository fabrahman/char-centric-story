import numpy as np
import re
import json
import sys, os

#sys.path.append('.')
#sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
path = os.path.dirname(os.path.abspath(__file__))


def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio

data = [json.loads(line.strip()) for line in open(os.path.join(path, "test_prediction_beams5_maxlen_15.jsonl"))]
correct_ans = []
wrong_ans = []
for i, item in enumerate(data):
    try:
        choices = re.search(r'\[choices\] (.*?) \[desc\]', item["input"].replace("\n", "")).group(1).split(", ")
    except:
        print(i)
    gold = item["gold"].strip("<eos>").strip()
    pred = item["predictions"].strip("<eos>").strip()

    scores = [levenshtein_ratio_and_distance(x, pred, ratio_calc = True) for x in choices] 

    max_ind = np.argmax(np.array(scores))
#    import pdb; pdb.set_trace()
    ans = int(choices[max_ind] == gold)
    correct_ans.append(ans)
    if ans != 1:
        wrong_ans.append(i)



print("accuracy is: {}".format(sum(correct_ans)/len(data)))
#print(wrong_ans)

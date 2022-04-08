import math

'''Defining the matrix A of percentage votes for each state/year'''
A = [
    [53.4, 59.1, 67.6, 60.3, 63.2],
    [52.4, 51.5, 57.8, 51.4, 60.5],
    [39.4, 42.5, 45.9, 45.2, 52.0],
    [42.9, 35.6, 56.8, 43.9, 51.3],
    [50.6, 52.5, 65.7, 55.5, 61.5]
]

'''Computing average of 2019 results'''
x = (50.5+45.6)/2
'''Creating centered vector of 2019 results'''
u = [0, 0, 50.5-x, 0, 45.6-x] 

'''Centering A to define a new matrix B'''
B = list(map(lambda row : \
             list(map(lambda a : \
                      a - sum(row)/len(row), row)), A))

'''Scalar product : sums the pairwise multiplication of vectors u and v'''
def sp(u, v):
    sp = 0

    for i in range(len(u)):
        sp = sp + u[i]*v[i]
    
    return sp;

'''Norm : square root of the scalar product of the vector u with itself'''
def norm(u):
    return math.sqrt(sp(u, u));

'''Cosine similarity : using formula given on the ICA sheet'''
def sim(u, b):
    return sp(u, b) / (norm(u) * norm(b))

'''kNN : predicts the percentage of votes for a state in 2019 using formula given on the ICA sheet'''
def predict(state, k):
    numerator = 0
    denominator = 0
    
    for i in range(k):
        s = C[i][len(C[i])-1]; 

        numerator = numerator + A[i][state]*s
        denominator = denominator + s

    return numerator / denominator;

'''Calculating the cosine similarity for each row and appending it to the end of the row'''
for i in range(len(A)):
    s = sim(u, B[i])
    
    A[i].append(s)
    B[i].append(s)

'''Sorting the rows by their cosine similarity (most similar -> least similar)'''
C = sorted(B, key = lambda row : \
           row[len(row)-1], reverse = True)
A = sorted(A, key = lambda row : \
           row[len(row)-1], reverse = True)

'''Creating an array to store the predictions and a filter indicating what states to predict for'''
predictions = []
'''0: Darlington, 1: Hartlepool, 3: Redcar'''
filterArr = [0, 1, 3] 

'''For each value of k (from 1 to 5) make the prediction for each state if it is in the filter array'''
for k in range(1, 6):
    row = []

    for state in range(len(C)):
        if(state in filterArr):
            row.append(predict(state, k))

    predictions.append(row)

'''From here downwards the only thing handled is the outputting of the predictions'''
header = ["k-value", "Darlington", "Hartlepool",
          "Redcar"]
lines = [header]

'''Formatting the predictions so they can be outputted in a nice arrangement'''
for j in range(len(predictions)):
    line = list(map(lambda p : \
                    str(round(p, 2)), predictions[j]))
    line.insert(0, j+1)
    
    lines.append(line)

'''Printing the predictions in a nice arrangement'''
print('kNN predictions of percentage of votes for \
2019 (%)')
for l in range(len(lines)):
    print('%-15s'*(len(header)) % tuple(lines[l]))












"""1. Implement and demonstratetheFIND-Salgorithm for finding the most specific 
hypothesis based on a given set of training data samples. Read the training data from a 
.CSV file and show the output for test cases. Develop an interactive program by 
Compareing the result by implementing LIST THEN ELIMINATE algorithm."""


#Find s algorithm then list eliminate

import pandas as pd

training_data = pd.read_csv('Book1.csv')

def find_s_algorithm(training_data):
    # Initialize the most specific hypothesis
    hypothesis = training_data.iloc[0, :-1].tolist()
    print(hypothesis)

    # Iterate over each training example
    for i in range(1, len(training_data)):
        instance = training_data.iloc[i, :-1].tolist()
        label = training_data.iloc[i, -1]
       

        # Check if the instance is positive
        if label == 'Yes':
            # Refine the hypothesis
            for j in range(len(hypothesis)):
                if instance[j] != hypothesis[j]:
                    hypothesis[j] = '?'

    return hypothesis

def list_then_eliminate_algorithm(training_data):
    # Initialize the general hypothesis space
    hypothesis_space = []

    # Iterate over each training example
    for i in range(len(training_data)):
        instance = training_data.iloc[i, :-1].tolist()
        label = training_data.iloc[i, -1]

        # Check if the instance is positive
        if label == 'Yes':
            # Eliminate inconsistent hypotheses from the hypothesis space
            hypothesis_space = [h for h in hypothesis_space if all(h[j] == instance[j] or h[j] == '?' for j in range(len(h)))]

            # Add the instance to the hypothesis space
            hypothesis_space.append(instance)

    # Select the most specific hypothesis from the remaining space
    hypothesis = ['?' for _ in range(len(training_data.columns) - 1)]
    for j in range(len(hypothesis)):
        values = set([h[j] for h in hypothesis_space])
        if len(values) == 1:
            hypothesis[j] = values.pop()

    return hypothesis

# Read the training data from CSV


# Apply FIND-S algorithm
find_s_hypothesis = find_s_algorithm(training_data)
print("Hypothesis using FIND-S algorithm:", find_s_hypothesis)

# Apply LIST THEN ELIMINATE algorithm
lte_hypothesis = list_then_eliminate_algorithm(training_data)
print("Hypothesis using LIST THEN ELIMINATE algorithm:", lte_hypothesis)

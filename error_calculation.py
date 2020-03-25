import numpy as np
from statistics import mean, stdev
import matplotlib.pyplot as plt
from sequential_classifier import sequential_classifier

def calculate_error(A, B, discriminants, ab_true, ba_true):
    J = 5
    K = 20
    total_error = []
    for i in range(1,J):
        for k in range(1,K):
            error_rate = 0
            classified_a = np.zeros(len(A))
            classified_b = np.zeros(len(B))
    
            # Sequential Classifier
            # seq_classifier = sequential_classifier.perform_classification(J=5) # need method from abel

            # Iterate through points in A
            for i, pt in enumerate(A):
                # classified_a[pt] = classifyPoints(J, pt[0], pt[1], seq_classifier[0], seq_classifier[1], seq_classifier[2]) # need method from abel
                
                # Add to error rate if class A is missclassified as class B
                if classified_a[pt] == 2:
                    error_rate += 1

            # Iterate through points in B
            for i, pt in enumerate(B):
                # classified_b[pt] = classifyPoints(J, pt[0], pt[1], seq_classifier[0], seq_classifier[1], seq_classifier[2]) # need method

                # Add to error rate if class B is missclassified as class A
                if classified_b[pt] == 1:
                    error_rate += 1

            total_error[i] = error_rate/200
        
        # a) average error rate
        average_error_rate = mean(total_error)
        # b) minimum error rate
        min_error_rate = min(total_error)
        # c) maximum error rate
        max_error_rate = max(total_error)
        # d) standard deviation of error rates
        stdev_error_rate = stdev(total_error)

        calulated_error_rates = [average_error_rate, min_error_rate, max_error_rate, stdev_error_rate]
        
        # Plot Error Rates
        J_vals = [1,2,3,4,5]

        plt.figure()
        plt.subplot(1,2,1)
        plt.title("Error Rate of Sequential Classifier as a function of J")
        plt.errorbar(J_vals, average_error_rate, stdev_error_rate, 'r', label='Avg Error Rate')
        plt.plot(J_vals, min_error_rate, 'b', label='Min Error Rate')
        plt.plot(J_vals, max_error_rate, 'g', label='Max Error Rate')
        plt.xlabel('J')
        plt.ylabel('Error Rate')
        plt.subplot(1,2,2)
        plt.title("Standard Deviation Error Rate of Sequential Classifier as a function of J")
        plt.plot(J_vals, stdev_error_rate, 'c', label='Stdev Error Rate')
        plt.xlabel('J')
        plt.ylabel('Error Rate')
        plt.tight_layout()
        plt.show()

    return calulated_error_rates
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:43:38 2024

@author: efthi
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import pandas as pd



import numpy as np

def perturb_data(X, epsilon=0.3, feature_idx=None, seed=None):
    """
    Apply random perturbations to the test data with a fixed random seed for reproducibility.
    :param X: Input features.  
    :param epsilon: Perturbation magnitude.
    :param feature_idx: Optional list of indices to perturb specific features.
    :param seed: Optional random seed for reproducibility.
    :return: Perturbed data.
    """
    if seed is not None:
        np.random.seed(seed)  # Set the seed for reproducibility

    X_adv = np.copy(X)
    n_samples, n_features = X.shape
    
    # Perturb either specific feature indices or all features
    if feature_idx is None:
        feature_idx = range(n_features)
    
    for i in range(n_samples):
        for j in feature_idx:
            perturbation = np.random.uniform(-epsilon, epsilon)  # Small random perturbation
            X_adv[i, j] += perturbation
            # Ensure that the perturbed value remains within valid feature bounds (e.g., non-negative)
            X_adv[i, j] = np.clip(X_adv[i, j], 0, None)  # Clip to ensure valid range (optional)
    
    return X_adv


df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\datasets_smc_mcmc_CART\parkinsons.csv")
y = df.Target
X = df.drop(['Target'], axis=1)
X = X.to_numpy()
y = y.to_numpy()

random_states = 5
tree_states = 10
for k in range (random_states):
# Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=k)
    
    leaf = [1, 10, 20, 30]
    
    import matplotlib.pyplot as plt
    
    # List to store results for plotting
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    # Lists to store results for plotting
    initial_accuracies = []
    adv_accuracies = []
    avg_tree_sizes = []
    
    #for k in leaf:
    
        
    for tree_state in range(tree_states):
        states_acc = []
        states_ADV_acc = []
        # Train a Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=tree_state, max_depth=2, bootstrap=True)
        clf.fit(X_train, y_train)
        
        # Evaluate accuracy on test set before attack
        y_pred = clf.predict(X_test)
        initial_accuracy = accuracy_score(y_test, y_pred)
        
        # Generate adversarial examples by perturbing the test data
        epsilon = 3  # Amount of perturbation
        X_test_adv = perturb_data(X_test, epsilon=epsilon, seed=1)
        
        # Test Random Forest model on adversarial examples
        y_pred_adv = clf.predict(X_test_adv)
        adv_accuracy = accuracy_score(y_test, y_pred_adv)
        
        # Calculate average tree size
        tree_sizes = [tree.tree_.node_count for tree in clf.estimators_]
        avg_tree_size = np.mean(tree_sizes)
        
        # Store results
        avg_tree_sizes.append(f"{avg_tree_size:.2f}")
        initial_accuracies.append(initial_accuracy)
        adv_accuracies.append(adv_accuracy)
        states_acc.append(initial_accuracy)
        states_ADV_acc.append(adv_accuracy)
    print("random state: ", k," accuracy is: ", np.mean(states_acc))
    print("random state: ", k," adv accuracy is: ", np.mean(states_ADV_acc))
        
    #print(f"Min Samples Leaf: {k}")
print(f"Initial Accuracy: {initial_accuracy:.4f}")
print(f"Accuracy after Adversarial Attack: {adv_accuracy:.4f}")
print(f"Average Tree Size: {avg_tree_size:.2f}\n")

# Plotting the bar plots
x = np.arange(len(avg_tree_sizes))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width/2, initial_accuracies, width, label='Original Accuracy')
bar2 = ax.bar(x + width/2, adv_accuracies, width, label='Accuracy after Attack')

# Labels and title
ax.set_xlabel('Average Tree Size')
ax.set_ylabel('Accuracy')
ax.set_title('Original vs Adversarial Accuracy for Different Tree Sizes on Contraceptive')
ax.set_xticks(x)
ax.set_xticklabels(avg_tree_sizes)
ax.legend()

# Show values on top of bars
for b in bar1 + bar2:
    height = b.get_height()
    ax.annotate(f'{height:.2f}', xy=(b.get_x() + b.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.show()

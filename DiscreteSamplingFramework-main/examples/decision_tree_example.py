# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021

@author: efthi
"""
import copy
from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC

from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_wine
import numpy as np
from discretesampling.domain.decision_tree.helper_functions import *
from art.attacks import DecisionTreeAttack
from art.classifiers import SklearnClassifier



# wine = load_wine()
# X = wine.data
# y = wine.target
df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\datasets_smc_mcmc_CART\heart_disease.csv")
# df['Target'] = pd.Categorical(df['Target']).codes
# df['A'] = pd.Categorical(df['A']).codes

#df=df.drop(["Date"], axis = 1)
#df=df.drop(["month"], axis = 1)
#df=df.drop(["day"], axis = 1)
df = df.dropna()
y = df.Target
X = df.drop(['Target'], axis=1)
X = X.to_numpy()
y = y.to_numpy()


acc = []


# dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
# try:
#     treeSamples = dtMCMC.sample(500)

#     mcmcLabels = dt.stats(treeSamples, X_test).predict(X_test, use_majority=True)
#     mcmcAccuracy = [dt.accuracy(y_test, mcmcLabels)]
#     print("MCMC mean accuracy: ", (mcmcAccuracy))
# except ZeroDivisionError:
#     print("MCMC sampling failed due to division by zero")


lam = [5]
for l in lam: 
    for i in range(1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4)
        a = l
        #b = 5
        target = dt.TreeTarget(a)
        initialProposal = dt.TreeInitialProposal(X_train, y_train)
        dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
        try:
            treeSMCSamples, current_possibilities_for_predictions = dtSMC.sample(100, 25, a, resampling= "residual")#systematic, knapsack, min_error, variational, min_error_imp, CIR
        
            smcLabels = dt.stats(treeSMCSamples, X_test).predict(X_test, use_majority=True)
            smcAccuracy = [dt.accuracy(y_test, smcLabels)]
            print("SMC mean accuracy: ", np.mean(smcAccuracy))
            acc.append(smcAccuracy)
            
            
            
            # Step 2: Generate adversarial examples by perturbing the test data
            epsilon = 3  # Amount of perturbation
            X_test_adv = perturb_data(X_test, epsilon=epsilon, seed = 1)
    
            # Step 3: Test Random Forest model on adversarial examples
            y_pred_adv = dt.stats(treeSMCSamples, X_test_adv).predict(X_test_adv, use_majority=True)
            adv_accuracy = dt.accuracy(y_test, y_pred_adv)
    
            print(f"Accuracy after Gaussian Adversarial Attack (Perturbation-based): {adv_accuracy:.4f}")
        
        except ZeroDivisionError:
            print("SMC sampling failed due to division by zero")
        
print("overall acc for 10 mc runs is: ", np.mean(acc))

'''
previous position of the wrapper
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
'''
uncomment to create all the neccesary info for DTs
'''
for i in range(len(treeSMCSamples)):
    tree = restructure_tree(treeSMCSamples[i].tree)
    # Preprocess the tree to extract node information
    parent_nodes, child_nodes, max_node_id = preprocess_tree(tree)
    # Compute leaf nodes and threshold list
    Leaf_nodes_pr = get_leaf_nodes(parent_nodes, child_nodes)
    threshold_list, feature_list = get_threshold_and_feature_lists(tree, max_node_id)
    # Get the left and right child lists
    left_child, right_child = find_left_right_chilren(tree, max_node_id)
    
    values, unique_labels = compute_values(
        X_train, y_train, 
        feature_list, threshold_list, left_child, right_child
    )
    
    #Initialize and test the wrapper on the original tree
    wrapper_tree_unchanged = SMCWrapperTree(feature_list, threshold_list, left_child, right_child, values, unique_labels) # uncomment this
    #wrapper_tree.fit(X_train, y_train) #uncomment this
    wrapper_tree_unchanged.fit(X_train, y_train)
    probabilities = wrapper_tree_unchanged.predict_proba(X_test)
    predictions = wrapper_tree_unchanged.predict(X_test)
    
    #Predict on the test set
    y_pred_unchanged = wrapper_tree_unchanged.predict(X_test)
    #Evaluate accuracy
    
    #Test cross-validation
    # cv_scores = cross_val_score(wrapper_tree_unchanged, X, y, cv=9)
    
    fake_dt = FakeScikitlearnDecisionTreeClassifier(wrapper_tree_unchanged)
    
    from art.attacks import DecisionTreeAttack
    
    #clf_artRF = SklearnClassifier(wrapper_tree_unchanged)
    attack_SMC = DecisionTreeAttack(fake_dt)
    adv_RF_SMC = attack_SMC.generate(X_test)
    adv_predictions_RF_SMC = fake_dt.predict(adv_RF_SMC)
    
    unique_labels = np.unique(y_train)  # e.g. array([1, 2])
    pred_class_idx = np.argmax(adv_predictions_RF_SMC, axis=1)  # e.g. [1, 0, 1, 0]
    predicted_labels = unique_labels[pred_class_idx]  # e.g. [2, 1, 2, 1]
    adv_acc_RF = accuracy_score(y_test, predicted_labels)
    
    
    print("original acc RF: ", accuracy_score(y_test, y_pred_unchanged))
    print("adversarial acc RF: ", adv_acc_RF)







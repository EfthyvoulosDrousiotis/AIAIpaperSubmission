from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from art.attacks import DecisionTreeAttack
from art.classifiers import SklearnClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\datasets_smc_mcmc_CART\heart_disease.csv")
y = df.Target
X = df.drop(['Target'], axis=1)
X = X.to_numpy()
y = y.to_numpy()

tree_state = 10
num_of_states = 5
overall_acc = []
overall_acc_pertrubed = []


for i in range(num_of_states):    
    accuracies_per_state_pertrubed = []
    accuracies_per_state = []
    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    for k in range(tree_state):
        # Train a Decision Tree Classifier on the training set
        clf = DecisionTreeClassifier(random_state=k)
        clf.fit(X_train, y_train)
        
        # Wrap the trained classifier with ART
        clf_art = SklearnClassifier(clf)
        
        # Evaluate the classifier on the original test set
        original_predictions = clf.predict(X_test)
        original_accuracy = accuracy_score(y_test, original_predictions)
        
        
        # Generate adversarial examples using the DecisionTreeAttack on the test set
        attack = DecisionTreeAttack(clf_art)
        adv = attack.generate(X_test)
        
        # Evaluate the classifier on the adversarial examples
        adv_predictions = clf.predict(adv)
        adv_accuracy = accuracy_score(y_test, adv_predictions)
        accuracies_per_state_pertrubed.append(adv_accuracy)
        overall_acc_pertrubed.append(adv_accuracy)
        accuracies_per_state.append(original_accuracy)
        overall_acc.append(original_accuracy)
        
    print("original acc for state: ", i, " is: ", np.mean(accuracies_per_state))
    print("parturbed acc for state: ", i, " is: ", np.mean(accuracies_per_state_pertrubed))


print("===========================")
print("average original acc for state is: ", np.mean(overall_acc))
print("average parturbed acc for state is: ", np.mean(overall_acc_pertrubed))

# RF = RandomForestClassifier(n_estimators=100, random_state=0)
# RF.fit(X_train, y_train)

# # Evaluate accuracy on test set before attack
# y_pred = RF.predict(X_test)
# initial_accuracy = accuracy_score(y_test, y_pred)

# clf_artRF = SklearnClassifier(RF)
# attack_RF = DecisionTreeAttack(clf_art)
# adv_RF = attack.generate(X_test)
# adv_predictions_RF = RF.predict(adv_RF)
# adv_acc_RF = accuracy_score(y_test, adv_predictions_RF)


# print("original acc RF: ", initial_accuracy)
# print("adversarial acc RF: ", adv_acc_RF)




















# # Use an offset with the attack
# attack_with_offset = DecisionTreeAttack(clf_art, offset=0.01)
# adv_offset = attack_with_offset.generate(X_test)

# # Evaluate the classifier on the adversarial examples with offset
# adv_offset_predictions = clf.predict(adv_offset)
# adv_offset_accuracy = np.mean(adv_offset_predictions == y_test)
# print(f"Accuracy on adversarial examples with offset: {adv_offset_accuracy * 100:.2f}%")



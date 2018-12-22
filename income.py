import pandas as pd 
import numpy as np
from math import log2 as log
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from random import sample

f = open('output.txt', 'a+')

def read_files():
    train_data = pd.read_csv("income.train.txt",header=None)
    dev_data = pd.read_csv("income.dev.txt",header=None)
    test_data = pd.read_csv("income.test.txt",header=None)
    train_data.columns = ["age", "work_class", "education", "marital_status", "occupation","race", "sex", "hours", "country", "income"]
    dev_data.columns = ["age", "work_class", "education", "marital_status", "occupation","race", "sex", "hours", "country", "income"]
    test_data.columns = ["age", "work_class", "education", "marital_status", "occupation","race", "sex", "hours", "country", "income"]
    return train_data,dev_data,test_data

def null_check(value):
    if value == 0:
        return 0
    else:
        return value

def null_check_log(value):
    if value == 0:
        return 0
    else:
        return log(value)

def calculate_probability(value,total_count):
    probability = null_check(value/total_count)
    return probability

def calculate_positive_negative(dataframe,column,value):
    target = dataframe.iloc[:,-1]   
    if value is None:
        positive = len(dataframe[(target == " >50K")])
        negative = len(dataframe[(target == " <=50K")])
    else:
        positive = len(dataframe[(column == value) & (target == " >50K")])
        negative = len(dataframe[(column == value) & (target == " <=50K")])
    return positive,negative

def calculate_threshold(column):
    sorted_column = column.sort_values()
    sorted_column = list(sorted_column)
    thresholds = []
    for i in range(len(sorted_column)-1):
        f = sorted_column[i] + ((sorted_column[i+1] - sorted_column[i])/2)
        thresholds.append(f)
    return thresholds

def find_unique_values(column):
    data_type = column.dtype
    if data_type == np.object:
        return np.unique(column),True
    else:
        return np.unique(calculate_threshold(column)),False
        
def calculate_entropy(positive,negative):
    total_data_count = positive + negative
    positive_probability = calculate_probability(positive,total_data_count)
    negative_probability = calculate_probability(negative,total_data_count) 
    positive_entropy = positive_probability*null_check_log(positive_probability)
    negative_entropy = negative_probability*null_check_log(negative_probability)
    entropy = - positive_entropy - negative_entropy
    return entropy

def calculate_information_gain(dataframe,column):
    unique_values,is_discrete = find_unique_values(column)
    entropy_values = []
    postitive_total,negative_total = calculate_positive_negative(dataframe,column,None)
    total_data_count = len(column)
    entropy_total = calculate_entropy(postitive_total,negative_total)
    all_continuous_gain = []
    best_threshold = None
    if is_discrete is True:
        for value in unique_values:
            positive, negative = calculate_positive_negative(dataframe,column,value) 
            count_value = positive + negative
            entropy = calculate_probability(count_value,total_data_count)*calculate_entropy(positive,negative)
            entropy_values.append(entropy)
        information_gain = entropy_total - np.sum(entropy_values)
    else:
        modified_column = pd.DataFrame(columns=["temp"])
        for threshold in unique_values:
            modified_column = column <= threshold 
            for value in [True,False]:
                positive, negative = calculate_positive_negative(dataframe,modified_column,value)           
                count_value = positive + negative
                if count_value == 0:
                    entropy = 0
                else:
                    entropy = calculate_probability(count_value,total_data_count)*calculate_entropy(positive,negative)
                entropy_values.append(entropy)
            information_gain = entropy_total - np.sum(entropy_values)
            entropy_values = []
            all_continuous_gain.append(information_gain)
        maximum_index = np.argmax(all_continuous_gain)
        best_threshold = unique_values[maximum_index]
        information_gain = all_continuous_gain[maximum_index]
    return information_gain,best_threshold

def maximum_information_gain(dataframe,feature_vector_attributes):
    all_information_gain = []
    threshold_value = None
    for fv in feature_vector_attributes:        
        information_gain,threshold = calculate_information_gain(dataframe,dataframe[fv])
        if threshold != None:
            threshold_value = threshold
        all_information_gain.append(information_gain)
    maximum_index = np.argmax(all_information_gain)
    node_to_split = list(feature_vector_attributes)[maximum_index]  
    return node_to_split,threshold_value

class node_details:
    nodes = 0
    def __init__(self,node_name,node_label,node_is_leaf,positive,negative,threshold,children):
       self.name = node_name
       self.label = node_label
       self.is_leaf = node_is_leaf
       self.positive = positive
       self.negative = negative
       self.threshold = threshold
       self.children = {}

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

def decision_tree(dataframe,feature_vector_attributes):
    node = node_details(None,None,None,None,None,None,None)
    node.positive , node.negative = calculate_positive_negative(dataframe,None,None)
    node.is_leaf = False

    if node.positive > node.negative:
            node.label = " >50K"
    else:
        node.label = " <=50K"

    if node.negative == 0:
        node.label = " >50K"
        node.is_leaf = True
        return node

    if node.positive == 0:
        node.label = " <=50K"
        node.is_leaf = True
        return node

    if len(feature_vector_attributes)==0:
        node.is_leaf = True
        if node.positive > node.negative:
            node.label = " >50K"
        else:
            node.label = " <=50K"
        return node
    
   
    best_node,threshold = maximum_information_gain(dataframe,feature_vector_attributes)
    node_values,_ = find_unique_values(dataframe[best_node])
    node.name = best_node
    new_attributes = set(feature_vector_attributes)
    new_attributes.discard(best_node)    

    if node_values.dtype == np.object:
        node.threshold = None
        for value in node_values:
            node.children[value] = None
            dataframe_subset = dataframe[dataframe[best_node] == value]
            if len(dataframe_subset)==0:
                node.is_leaf = True
                return node
            else: 
                node.children[value]  = decision_tree(dataframe_subset,new_attributes)    
    else:
        node.threshold = threshold
        dataframe_subset_less = dataframe[dataframe[best_node] <= threshold]
        value_less = "<="+str(threshold)
        if len(dataframe_subset_less)==0:
            node.is_leaf = True
            return node
        else:
            node.children[value_less]  = decision_tree(dataframe_subset_less,new_attributes)
    return node

def testing_decision_tree(tree,data):
    while tree.is_leaf == False:
        node_name = tree.name
        data_in_node = data[node_name]
        if tree.threshold == None:
            if data_in_node in tree.children.keys():
                tree = tree.children[data_in_node]
            else:
                if tree.positive > tree.negative:
                    return " >50K"
                else:
                    return " <=50K"
        else:
            tree = tree.children["<="+str(tree.threshold)]
    return tree.label

def accuracy(tree,data,type_of_data):
    target = data.iloc[:,-1]
    data = data.drop('income',1)
    number_of_rows = data.shape[0]
    number_of_correct_predictions = 0

    for index in range(number_of_rows):
        data_to_traverse = data.iloc[index]
        target_to_check = target.iloc[index]
        prediction = testing_decision_tree(tree,data_to_traverse)
        if prediction == target_to_check:
            number_of_correct_predictions+=1
    
    accuracy = number_of_correct_predictions/number_of_rows
    print("Accuracy of %s data using Decision tree from Scratch: %f" %(type_of_data, accuracy*100),file = f)

def preprocessing_sklearn(training_data,development_data,testing_data):
    train_label = training_data.iloc[:,-1]
    dev_label = development_data.iloc[:,-1]
    test_label = testing_data.iloc[:,-1]

    train_data = training_data.drop('income',1)
    dev_data = development_data.drop('income',1)
    test_data = testing_data.drop('income',1)

    combined_dataset = pd.concat([train_data,dev_data,test_data],keys=['train','dev','test'])
    one_hot_encoded_data  = pd.get_dummies(combined_dataset)
    train_data, dev_data, test_data = one_hot_encoded_data.xs('train'), one_hot_encoded_data.xs('dev'), one_hot_encoded_data.xs('test')
    
    return train_data,train_label,dev_data,dev_label,test_data,test_label

def decision_tree_sklearn(training_data,testing_data,train_label,test_label,type_of_data):
    model = DecisionTreeClassifier().fit(training_data, train_label)
    predicted = model.predict(testing_data)
    accuracy = accuracy_score(test_label,predicted)
    print("Accuracy of %s data using Decision tree from Sklearn: %f" %(type_of_data, accuracy*100),file = f)

train_data,dev_data,test_data = read_files()
attributes = ['age', 'work_class', 'education', 'marital_status', 'occupation','race', 'sex', 'hours', 'country']

tree = decision_tree(train_data,attributes)
train_data_sk,train_label_sk,dev_data_sk,dev_label_sk,test_data_sk,test_label_sk = preprocessing_sklearn(train_data,dev_data,test_data)

print(file =f)
print("Decision tree:", file =f)
print(file =f)
accuracy(tree,train_data,"training")
decision_tree_sklearn(train_data_sk,train_data_sk,train_label_sk,train_label_sk,"training")
print(file =f)
accuracy(tree,dev_data,"development")
decision_tree_sklearn(train_data_sk,dev_data_sk,train_label_sk,dev_label_sk,"development")
print(file =f)
accuracy(tree,test_data,"testing")
decision_tree_sklearn(train_data_sk,test_data_sk,train_label_sk,test_label_sk,"testing")

f.close()
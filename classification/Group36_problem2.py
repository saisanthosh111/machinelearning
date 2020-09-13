import pandas as pd
import numpy as np
import math

# calculating the mean
def mean(data):
    return np.mean(data)
# calculating the standard deviation
def std(data):
    return np.std(data)

def summary(data):
    for i in zip(*data):
        yield {
                'mean': mean(i),
                'stdev': std(i)
            }

# calculating the prior probabilities
def priorprobability(feature_data,data):
    return float(len(feature_data))/float(len(data))

# summarising the data
def train_dataset(data_seperated,data):
    featured_data = {}
    for i in data_seperated:
        featured_data[i]={
            'prior_probability':priorprobability(data_seperated[i],train_data),
            'summary_data':[i for i in summary(data_seperated[i])],
        }
    return featured_data

# calculating the normal probability and returning it
def normal_probability(x,mean,stdev):
        exponent_power = -((x - mean) ** 2) / (2 * (stdev**2))
        exponent = math.e ** exponent_power
        denominator = ((2 * math.pi) ** .5) * stdev
        normal_prob = exponent / denominator
        return normal_prob

# calculating the joint probability and returning it
def joint_probability(featured_data,test_row):
    required_joint = {}
    for target_label,features in featured_data.items():
        likelihood = 1
        for j in range(len(features['summary_data'])):
            feature = test_row[j]
            mean1    = features['summary_data'][j]['mean']
            stdev1   = features['summary_data'][j]['stdev']
            normal_prob = normal_probability(feature,mean1,stdev1)
            likelihood = likelihood*normal_prob
        prior_probability1 = features['prior_probability']
        required_joint[target_label] = prior_probability1*likelihood
    return required_joint

def marginal_probability(joint_probabilities):
    return sum(joint_probabilities.values())

# calculating the posterior probabilities and returning the dict of it
def posterior_probability(featured_data,test_row):
    posterior_probabilities = {}
    joint_probabilities = joint_probability(featured_data,test_row)
    marginal_probability1 = marginal_probability(joint_probabilities)
    for target_label,joint_prob in joint_probabilities.items():
        posterior_probabilities[target_label] = joint_prob/marginal_probability1
    return posterior_probabilities


# calculating the maximum posterior probability and returning it
def posterior_maximum(featured_data,test_row):
    posterior_probabilities = posterior_probability(featured_data,test_row)
    return max(posterior_probabilities,key=posterior_probabilities.get)

# prediction the class label for the test data and returning it
def prediction_test(featured_data,test_data):
    feature_labels = []
    for row in test_data.values:
        feature_labels.append(posterior_maximum(featured_data,row))
    return feature_labels

# grouping the data based on class
def categorise(data):
    return data.groupby('class')[['sepallength', 'sepalwidth','petallength','petalwidth']].apply(lambda g: g.values.tolist()).to_dict()

# calculating the accuracy
def calculate_accuracy(test_data_set,featured_data):
    num_correct = 0
    actual_prediction = [i[-1] for i in test_data_set.values]
    predicted_data = prediction_test(featured_data,test_data_set)
    for a,b in zip(actual_prediction,predicted_data):
        if a==b:
            num_correct += 1
    return num_correct/float(len(test_data_set))*100

if  __name__=="__main__":
    # loading the data
    train_data = pd.read_csv("D:/train.csv",names=["sepallength","sepalwidth","petallength","petalwidth","class"],header=0)
    test_data = pd.read_csv("D:/test.csv",names=["sepallength","sepalwidth","petallength","petalwidth","class"],header=0)
    # grouping the data based on class
    seperated_data = categorise(train_data)
    # summarising the data set
    featured_data = train_dataset(seperated_data,train_data)
    print("The accuracy of naive bayas classfier is {} %".format(calculate_accuracy(test_data,featured_data)))
#

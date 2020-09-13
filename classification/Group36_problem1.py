import pandas as pd
import numpy as np
import math

def summary(data):
    for i in zip(*data):
        yield {
                'mean': mean(i),
                'stdev':std(i)
            }

# calculating the mean data
def mean(data):
    return np.mean(data)

# calculating the standard deviation
def std(data):
    return np.std(data)

# calculating the aprori probability
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

# calculating the marginal probability

def marginal_probability(joint_probabilities):
    return sum(joint_probabilities.values())


def covarince_construction(p,p1,p2,p3):
    return np.cov([p,p1,p2,p3],bias=True)

# constructing the covariance matrix an returning it
def covariance_matrix(seperated_data,class_wanted):
    p=[]
    p1=[]
    p2=[]
    p3=[]
    for i in seperated_data[class_wanted]:
        p.append(i[0])
        p1.append(i[1])
        p2.append(i[2])
        p3.append(i[3])
    cov_matrix = covarince_construction(p,p1,p2,p3)
    return cov_matrix

# constructing a list of means of the same class but different features
def mean_construction(featured_data,class_wanted):
    mean_wanted=[]
    for i,j in featured_data.items():
        if i==class_wanted:
            for k in featured_data[i]['summary_data']:
                mean_wanted.append(k["mean"])
        else:
            continue
    return mean_wanted

# calculating the multivariate normal distribution
def norm_pdf_multivariate(x, mu, sigma):
    mu = np.array(mu)
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = np.linalg.inv(sigma)
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")


# calculating the posterior probabilities and returning it
def posterior_probabilities(test_row,featured_data,seperated_data):
    joint = {}
    posterior = {}
    for target_label,features in featured_data.items():
        mean = mean_construction(featured_data,target_label)
        variance = covariance_matrix(seperated_data,target_label)
        likelihood = norm_pdf_multivariate(test_row,mean,variance)
        joint_probability = features['prior_probability']*likelihood
        joint[target_label] = joint_probability
    marginalprobability = marginal_probability(joint)
    for target_label,joint_prob in joint.items():
        posterior[target_label]=joint_prob/marginalprobability
    return posterior

# checking the probabilities and returning the maximum probability label
def posterior_maximum_base(test_row,featured_data,seperated_data):
    maximum = 0
    posterior_probability = posterior_probabilities(test_row,featured_data,seperated_data)
    return max(posterior_probability,key=posterior_probability.get)

#expecting the label on the basis of features and returning it
def bayas_probability(featured_data,test_data,seperated_data):
    feature_labels = []
    for row in test_data.values:
        feature_labels.append(posterior_maximum_base(row[0:4],featured_data,seperated_data))
    return feature_labels

# categorising the data on the basis of class
def categorise(data):
    return data.groupby('class')[['sepallength', 'sepalwidth','petallength','petalwidth']].apply(lambda g: g.values.tolist()).to_dict()

# calculating the accuracy for bayas classifier
def calculate_accuracy_base(test_data_set,featured_data,seperated_data):
    num_correct = 0
    actual_prediction = [i[-1] for i in test_data_set.values]
    predicted_data = bayas_probability(featured_data,test_data_set,seperated_data)
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
    #summary of each feature
    featured_data = train_dataset(seperated_data,train_data)
    # calculating the accuracy for bayas classifier
    p = calculate_accuracy_base(test_data,featured_data,seperated_data)
    print("The accuracy of bayas classfier is {} percent".format(p))

"""
@author: rmisra
Handwritten digit classification
"""
import numpy
import math
import matplotlib.pylab as py

# custom sigmoid function; if -x is too large, return value 0
def sigmoid(x):
    if(-x < 710):
        return 1 / (1 + math.exp(-x))
    else:
        return 0.0

numpy.random.seed(0)

# Training on 3's and 5's dataset using Newton's method 
train3 = open("train3.txt",'r')
bitRepresentation3 = [list(map(float,line.strip('\n').split())) for line in train3.readlines()]
train3.close()

train5 = open("train5.txt",'r')
bitRepresentation5 = [list(map(float,line.strip('\n').split())) for line in train5.readlines()]
train5.close()

# training set
bitImage = bitRepresentation3 + bitRepresentation5
# labeling 3's class as 1 and 5's class as 0
label = [1]*len(bitRepresentation3) + [0]*len(bitRepresentation5)

t = len(bitImage)
weights = numpy.matrix(numpy.zeros(64))


log_likehood_plot = numpy.zeros(20)
error_plot = numpy.zeros(20)

for i in range(0,20):
    # initialise Hessian and Gradient
    Hessian = numpy.matrix(numpy.zeros((64,64)))
    gradient = numpy.matrix(numpy.zeros(64))

    # calculate gradient and Hessian matrix over all the samples
    for j in range(0,t):
        gradient += (label[j] - sigmoid(numpy.dot(weights, bitImage[j])))*numpy.array(bitImage[j])
        
        Hessian += sigmoid(numpy.dot(weights, bitImage[j])) * sigmoid(-1*numpy.dot(weights, bitImage[j]))*(numpy.transpose(numpy.matrix(bitImage[j]))* numpy.matrix(bitImage[j]))
    
    # update weights vector according to the update rule of Newton's method
    weights = weights + numpy.transpose(numpy.linalg.inv(Hessian) * numpy.transpose(gradient))
    
    # calculating log likelhood
    log_likelihood = 0.0
    for j in range(0,t):
        log_likelihood += (label[j]*numpy.log(sigmoid(numpy.dot(weights, bitImage[j])))) + ((1-label[j])*numpy.log(sigmoid(-1*numpy.dot(weights, bitImage[j]))))
        
    log_likehood_plot[i] = log_likelihood

    # calculating error percentage
    error = 0.0;    
    for j in range(0,t):
        prediction = sigmoid(numpy.dot(weights, bitImage[j]))
        if(prediction>0.5 and label[j]!=1):
            error += 1;
        elif (prediction<=0.5 and label[j]!=0):
            error += 1;
            
    error_plot[i] = error*100/t;

# ploting log likelihood
py.plot(log_likehood_plot, 'bo')
py.tick_params(labelright = True)
py.xlabel('Iteration')
py.ylabel('Log Likelihood')
py.title("Iteration vs Log Likelihood")
py.savefig("Log Likelihood Handwritten digit.pdf")
py.show()

# ploting error percentage
py.plot(error_plot, 'bo')
py.tick_params(labelright = True)
py.xlabel('Iteration')
py.ylabel('Error Percentage')
py.title("Iteration vs Error Percentage")
py.savefig("Error Handwritten digit.pdf")
py.show()


print('Optimal log likelihood : ' + str(log_likehood_plot[len(log_likehood_plot)-1]));
print('Optimal Weights : \n' + str(numpy.reshape(numpy.matrix(weights),(8,8))));
                                          
# calculating error on test dataset
test3 = open("test3.txt",'r')
bitRepresentation3 = [list(map(float,line.strip('\n').split())) for line in test3.readlines()]
test3.close()

test5 = open("test5.txt",'r')
bitRepresentation5 = [list(map(float,line.strip('\n').split())) for line in test5.readlines()]
test5.close()

# testing set
bitImage = bitRepresentation3 + bitRepresentation5
# labeling 3's class as 1 and 5's class as 0
label = [1]*len(bitRepresentation3) + [0]*len(bitRepresentation5)
t = len(bitImage)

error = 0.0;    
for j in range(0,t):
    prediction = sigmoid(numpy.dot(weights, bitImage[j]))
    if(prediction>0.5 and label[j]!=1):
        error += 1;
    elif (prediction<=0.5 and label[j]!=0):
        error += 1;

# printing error on training and testing dataset
print('Error on training dataset : ' + str(error_plot[len(error_plot)-1]) + '%');
print('Error on test dataset : ' + str(error*100/t) + '%');

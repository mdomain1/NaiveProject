

import math

#Constants defined here
NUM_TRAINING_IMGS = 5000
NUM_TESTING_IMGS = 1000
NUM_CLASSES = 10
NUM_FEATURES = 784
IMG_DIM = 28  #28x28 digit images

#initialize files for training images/labels
filename = 'trainingimages'
f = open(filename,'r')
training_images = f.read()
training_images = training_images.replace('\n','') #strip new lines


filename = 'traininglabels'
f = open(filename,'r')
training_labels = f.read()
training_labels = training_labels.replace('\n','') #strip new lines

#initialize files for training images/labels
filename = 'testimages'
f = open(filename,'r')
test_images = f.read()
test_images = test_images.replace('\n','') #strip new lines

filename = 'testlabels'
f = open(filename,'r')
test_labels = f.read()
test_labels = test_labels.replace('\n','') #strip new lines

def printDigit(index):
	for i in range(index*IMG_DIM,index*IMG_DIM+IMG_DIM):
		for j in range(0,IMG_DIM):
			print(test_images[calcIndex(i,j)],end=" ")
		print("")


def calcIndex(row,col):
	return row*IMG_DIM + col

# Inputs: ldict- likelihood diciontary, n- num features , X - num classes
# Function: inits a lookup table with X keys each containng a list of size n
def init_likelihood(ldict,n,X):
	for i in range(0,X):
		ldict[i] = []
		for j in range(0,n):
			ldict[i].append(0)

def init_posterior(dict,X):
	for i in range(0,X):
		dict[i] = []


# inputs: ftable - table to init, X - num classes
# Function: inits a table with X keys to 0
def init_freq_table(ftable,X):
	for i in range(0,X):
		ftable[i] = 0   


# Laplace Smoothing 
# Inputs: likelihood-Likelihood calculation thus far,ftable- freq table to finish likelihood calculations,
#         priors- prior table, k- constant to add, V- # of possible values the features can take on
# Function- performs Laplacian smoothing and finishes calculating likelihoods/priors
def smooth(likelihood,ftable,priors,k,V):
	for key in range(0,NUM_CLASSES):
		priors[key] = ftable[key]/NUM_TRAINING_IMGS
		for i in range(0,IMG_DIM*IMG_DIM):
			likelihood[key][i] = (likelihood[key][i]+k)/(ftable[key]+k*V)
			likelihoodb[key][i] = (likelihoodb[key][i]+k)/(ftable[key]+k*V)


def naive_bayes_classifier(ftable,likelihood,priors,k,V):
	# Loop through every "pixel" of training data to calculate likelihood and frequency table
	for i in range(0,NUM_TRAINING_IMGS):
		label = int(training_labels[i]) #training labels are strings to convert to int
		ftable[label] = ftable[label] + 1 #update freq table for each digit

		for r in range(IMG_DIM*i,IMG_DIM*i + IMG_DIM):
			for c in range(0,IMG_DIM):
				pixel = training_images[calcIndex(r,c)]
				index = calcIndex(r-IMG_DIM*i,c)  #turn r back into 0-27
				if(pixel=='+' or pixel=='#'):
					likelihood[label][index] = likelihood[label][index] + 1
				else:
					likelihoodb[label][index] = likelihoodb[label][index] + 1

	smooth(likelihood,ftable,priors,k,V)

def MAP_classification(priors,likelihood,start,classifications,posterior_table):
	results = []
	for i in range(0,NUM_CLASSES):
		posterior = math.log(priors[i])
		for j in range(start,start+IMG_DIM*IMG_DIM):
				pixel = test_images[j]
				if(pixel=='+' or pixel=='#'):
					posterior = posterior + math.log(likelihood[i][j-start]) #j-start because we want index to be 0-783
				else:
					posterior = posterior + math.log(likelihoodb[i][j-start])
		results.append(posterior)
		posterior_table[i].append(posterior)
	classifications.append(results.index(max(results)))

def generate_confusion_matrix(confusion_dict):
	for key in range(0,NUM_CLASSES):
		print(key,end=" ")
		for value in confusion_dict[key]:
			confusion_entry = (value/test_labels.count(str(key)))*100
			print(round(confusion_entry,2),end = " ")
		print("")


# Test to find value of k that produces highest classification accuracy.

ktest = []
for k in range(1,10):
	# Data structures for Training
	likelihood = dict()
	likelihoodb=dict()
	digit_freq = dict() #keeps track of total # of training examples from this class
	priors = dict()

	###   INITIALIZATION   ###
	init_likelihood(likelihood,NUM_FEATURES,NUM_CLASSES)
	init_likelihood(likelihoodb,NUM_FEATURES,NUM_CLASSES)
	init_freq_table(digit_freq,NUM_CLASSES)
	init_freq_table(priors,NUM_CLASSES)

	###   TRAINING   ###
	V=2
	naive_bayes_classifier(digit_freq,likelihood,priors,k,V)

	classifications = [] # Classification results 
	posterior_table = dict() # Table to hold posteriors for evaluating prototypical instances of each digit class
	init_posterior(posterior_table,NUM_CLASSES)

	for i in range(0,NUM_TESTING_IMGS):
		start = IMG_DIM*IMG_DIM*i
		MAP_classification(priors,likelihood,start,classifications,posterior_table)
	print("Classifications : ", classifications)

	total_correct = 0 # Total number correct
	digits_correct = dict()
	init_freq_table(digits_correct,NUM_CLASSES)
	confusion_dict = dict() # 10 entries with a list of size 10 to hold the misclassifcations of each class
	init_likelihood(confusion_dict,NUM_CLASSES,NUM_CLASSES)

	for i in range(0,NUM_TESTING_IMGS):
		if(classifications[i]==int(test_labels[i])):
			total_correct = total_correct + 1
			digits_correct[int(test_labels[i])] = digits_correct[int(test_labels[i])] + 1

		confusion_dict[int(test_labels[i])][classifications[i]] = confusion_dict[int(test_labels[i])][classifications[i]]+1
		

	# Calculate classification rate for each digit - UNCOMMENT
	for i in range(0,NUM_CLASSES):
		digits_correct[i] = digits_correct[i]/test_labels.count(str(i))
	print("Classification rate for each digit:")
	print(digits_correct) #display classification rate for each digit

	# Test examples from each digit class with the highest and lowest posterior probabilities
	for i in range(0,NUM_CLASSES):
	    print("Highest posterior probability for "+str(i)+":")
	    printDigit(posterior_table[i].index(max(posterior_table[i])))
	    print("Lowest posterior probability for "+str(i)+":")
	    printDigit(posterior_table[i].index(min(posterior_table[i])))

	# Confusion matrix
	generate_confusion_matrix(confusion_dict)


	# ktest
	ktest.append(total_correct/1000)
	print(k,end=" ")
	print(":",end=" ")
	print(total_correct/1000)

	print("Overall classification accuracy:")
	print(total_correct/1000)


# def odds(c1,c2):

#key
#(0,-1): '+'
#([-1,-3)): ' '
#Anything smaller than -3: '-'
def display_feat_likelihoods(c1):
	for i in range(0,IMG_DIM):
		for j in range(0,IMG_DIM):
			l = math.log(likelihood[c1][calcIndex(i,j)])
			if l>-1:
				print('+',end="")
				
			elif l>-2:
				print(' ',end="")
				
			elif l>-3:
				print(' ',end="")
				
			else:
				print('-',end="")

		print("")

#key
#postive '+'
#close to 0 (-1,0): '0'
#negative '-'
def display_odds_ratios(c1,c2):
	for i in range(0,IMG_DIM):
		for j in range(0,IMG_DIM):
			l = math.log(likelihood[c1][calcIndex(i,j)]/likelihood[c2][calcIndex(i,j)])
			if l>0:
				print('+',end="")
				
			elif l>-1:
				print('0',end="")
				
			elif l>-1:
				print('0',end="")
				
			else:
				print('-',end="")

		print("")			



###   ODDS RATIOS   ###

display_feat_likelihoods(4)
print("")
display_feat_likelihoods(9)
print("")
display_odds_ratios(4,9)



print(ktest)
print("best choice of k is:")
print(ktest.index(max(ktest)) + 1)






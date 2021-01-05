# Databyte_Inductions

# All my answers are as below:-

Page 2:

Q1) Suppose a whole number 'W' completely divides ((2^64) + 1). Which of the following number is completely divisible by 'W'?
Ans.- ((2^192) + 1)

Q2) Find the odd one out among 835, 734, 642, 751, 853, 981, 532
Ans.- 751

Q3) Find the maximum number of triangles.
Ans.- 24

Q4) Study the pie chart below. Suppose the amount of Royalty paid by the publisher is Rs. 2,81,250 for an edition of 12,500 copies. What should be the selling price of the book if the publisher desires a profit of 5%?
Ans.- Rs. 157.50

Q5) Study the chart given below. Which of the two countries accounts for higher-earning out of Services and Miscellaneous put together?
Ans.- Both spend equal amounts

Q6) Find 1st, 2nd & 4th central moments of the messages received on 9 consecutive days: 15, 11, 9, 5, 18, 4, 15, 13, 17.
Ans.- Here, Number of observations = N = 9 
Mean = M = 11.8889
1st Central moment = ∑fi(xi-M)/N      = 0
2nd Central moment = ∑fi(xi-M)^2/N = 22.5432
3rd Central moment = ∑fi(xi-M)^3/N  = -46.5961
4th Central moment = ∑fi(xi-M)^4/N  = 940.1786

Q7) Calculate the Eigen Value(s) for the given matrix.
Ans.- λ1=3 λ2=(-sqrt(29)+9)/2 λ3=(sqrt(29)+9)/2  Final expression for eigen values is -(λ-3)*(λ^2-9*λ+13) = 0 [Space is less for further explanation]]

Q8) A biased dice with the property that the probability of a face with n dots showing up is proportional to n. The probability of face showing 3 dots is?
Ans.- Let P[1]=1a,P [2]=2a ... P[6]=6a 
The Sum of all the probabilities for an event is one.
So, ∑P(n) = 1
=>a = 1/21
Hence, P[3] = 3a = 1/7

Q9) In a locality, 33% go to work by Bike, 42% go by Car, and 12% use both. A random person skips work one day of the week. The probability that a random person went to work using neither of the above is? Explain your approach.
Ans.- Using Set Theory, we have 
Let B= percentage of people that go to work by Bike =33%=.33
C= percentage of people that go to work by Car = 42% = .42
So, B∩C= percentage of people that use both = 12% = .12
We know, 
BUC = percentage of people that use either one of them or both
         = B + C - B∩C 
         = .33 + .42 - .12 = .63
Now, (BUC)' = percentage of people that use none 
                      = 1 - BUC = 1 - .63 = .37
Hence, probability that a random person went to work using neither is = .37 = 37/100 

Q10) Three companies A, B and C supply 25%, 35% and 40% of the notebooks to a school. Past experience shows that 5%, 4% and 2% of the notebooks produced by these companies are defective. If a notebook was found to be defective, what is the probability that the notebook was supplied by A?
Ans.-  Let 
P(A) = the event that notebooks are provided by A = 0.25
P(B) = the event that notebooks are provided by B = 0.35
P(C) = the event that notebooks are provided by C = 0.4
and P(D)= the event that notebooks are defective
Then,
P(D|A) = 0.05 
P(D|B) = 0.04 
P(D|C) = 0.02
We know, by Bayes' theorem
P(A│D) = (P(D│A)*P(A))/(P(D│A) * P(A) + P(D│B) * P(B) + P(D│C) * P(C) )
              = (0.05*0.25)/((0.05*0.25)+(0.04*0.35)+(0.02*0.4)) 
              = 2000/(80*69) 
              = 25/69
Hence, probability that the defective notebook was supplied by A = 25/69

Q11) In a colony, there are 55 members. Every member posts a greeting card to all the members. How many greeting cards were posted by them?
Ans.- Each player will post greeting cards to the remaining 54 members in different ways. 
So, the total numbers of greeting cards posted are
54 + 54 + 54 … + 54(55times) = 54 x 55 = 2970

Q12) There are 20 points in a plane, how many triangles can be formed by these points if 5 are colinear?
Ans.- Total number of Points = n = 20
Number of Collinear Points = m = 5
We know,
Total number of triangles formed = (Triangles formed from all points) -  (Triangles formed from Collinear points)
                                                            = nC3 - mC3
                                                            = 20C3 - 5C3 = (20x19x18/3x2) - (5x4/2) 
                                                            = 1140-10
                                                            = 1130

Q13) Alice has 2 kids and one of them is a girl. What is the probability that the other child is also a girl? Explain your approach. (You can assume that there is an equal number of males and females in the world.)
Ans.- In the question, if we assume that the *Girl child is already born*, 
There are only two cases possible. 
1) the first child is a girl and the second child is a boy (gb)
2) the first child is a girl and the second child is a girl (gg)
So, the probability that the other child is also a girl = P(gg)=1/2

If we assume that there is no specific order of the child being born, then 
Three cases are possible. 
1) the first child is a girl and the second child is a boy (gb)
2) the first child is a girl and the second child is a girl (gg)
3) the first child is a boy and the second child is a girl (bg)
So, the probability that the other child is also a girl = P(gg)=1/3
Hence, based on the assumptions, the answer is either 1/2 or 1/3

Q14) In a class of 23 students, approximately what is the probability that two of the students have their birthday on the same day. Explain your approach.
Ans.- Assumption : Year is not a leap year( 1 year = 365 days )
We know, by the rule of complementary events, 
P(23 persons share birthday) = 1 - P(no two persons share birthday)

Calculating P(no two persons share birthday),
It is apparent that the first person can have any birthday. 
The second person's birthday has to be different. 
There are 364 different days to choose from, so the chance that two persons have different birthdays is 364/365. 
P(2 persons have different birthdays) = (365/365) * (364/365)

In the case of 3 persons, 
For the third person, there are 363 birthdays out of 365.
P(3 persons have different birthdays) = (365/365) * (364/365) * (363/365) 

Similarly for 4 persons, 
P(4 persons have different birthdays) =  (365/365) * (364/365) * (363/365) * (362/365)

So, we can conclude that for n persons,
P(n persons have different birthdays) = ((365-1)/365) * ((365-2)/365) * ((365-3)/365) * . . . * ((365-n+1)/365)
                                                = (365_P_n)/(365^n)
                                                = 365! / ((365-n)! * 365^n)

P(n persons share birthday) = 1 - P(n persons have different birthdays)
                                                   = 1 - 365! / ((365-n)! * 365^n)

Hence, P (23 persons share birthday) = 1-  [365! / {(365-23)! * 365^23}] 
                                                                   = 0.507297234  (Approx.) 

Q15) Hospital records show that 75% of patients suffering from a disease die due to that disease. What is the probability that 4 out of the 6 randomly selected patients recover? Explain your approach.
Ans.- The concept of binomial random variables can be used here because only 2 outcomes are possible here (dead or alive)

Here, 
Total number of trials = n = 6
Selections made = x = 4
Probability that selected patient is alive = p = 0.25
Probability that selected patient is dead = q = 0.75
P(4 out of the 6 randomly selected patients recover) 
         = (nCx)x(p^x)x(q^(n-x))
         = (6C4)x(0.25^4)x(0.75^(6-4))
         = 15 x 2.1973×10 ^(−3)
         = 0.0329595

Q16) Point out the correct statement
Ans.- Raw data is original source of data
A Straight Line of best fit for the following points (5, 14), (4, 13), (3, 11), (2, 8), (1, 6), (0, 3)
Ans.- y = 3.52 + 2.26x
Estimate the number of maggies sold in a month in India during the pre covid times. Take appropriate assumptions and explain your approach in detail.
Ans.- There are 1.38 billion people in the India, 
In India, 
Let 50% of its population be below the age of 25,
5% of population be above age of 65.

Now, 
Let's assume that all people below 25 eat maggie twice a day,
60% of people between 25-65 eat maggie once a day, 
and 10% of people above age of 65 eat maggie do not eat a day.

[These data are calculated as an average of the day considering total number of packets consumed in a 7 day week]

Calculation For a month (assuming 30 days month):
Total number of maggie packets sold
                                        = {(.50x1.0x2) +(.60x.45x1)+(.05x.10x0) }x1.38 billion
                                        ={1 + .27}x1.38 billion
                                        = 1.7526 Billion packets



# Page -3:

Q1) How will you explain Machine Learning to a 5-year-old?
Ans.-> Machine Learning is an application of artificial intelligence where we give access to data(information) so that they can learn the pattern and use that data for performing task( This task is usually something which a human or an intelligent animal can accomplish, such as learning, planning, problem-solving, etc.)  without explicitly being programmed to do so or give prediction

Q2) Each data point in a given dataset has a feature vector of N*1 where N is a large number. The classes are not well differentiable from the given dataset. How can we solve the problem?
Ans.-> The class is imbalanced in the given question. To deal with it, we’ve to improve classification algorithms or balance classes in the training data before providing the data as input to the machine learning algorithm. And, as it has a large dataset, to minimize loss we can use dimensionality reduction techniques. We can use this concept to reduce the number of features in your dataset without having to lose much information and keep (or improve) the model's performance

Q3) How can you deal with the problem of inadequate data in Machine Learning?
Ans.-> The simpler the machine learning algorithm, the better it will learn from small data sets. Small data requires models that have low complexity (or high bias) to avoid overfitting the model to the data. Naive Bayes algorithm is among the simplest classifiers and as a result learns remarkably well from relatively small data sets.
 
Q4) You are assigned a new project which involves helping a food delivery company save more money. The problem is, the company’s delivery team aren’t able to deliver food on time. As a result, their customers get unhappy. And, to keep them happy, they end up delivering food for free. Which machine learning algorithm can save them?
Ans.-> The given data is not enough to go for a machine learning approach. The pattern is missing. This is a route optimization problem

Q5) Your ML model returns poor accuracy during inference but during training, it gives accuracy as expected. What can be the reasons? How do you overcome them?
Ans.-> Overfitting might be the most common reason. To solve, we can scrap the current training dataset and collect a new training dataset.
Then we can re-split sample into train/test in a softer approach to getting a new training dataset.
It is possible that the training or test datasets are an unrepresentative sample of data from the domain. The remedy is often to get a larger and more representative sample of data from the domain.
 
Q6) You are given 10000 training data points unequally divided into more than 2 classes and 2000 testing data points. The tool has the following algorithms in-built: (a) SVM, (b) XGBoost (c) Neural Network. Which one will you use and why?
Ans.->  Clearly, here the dataset is very large. SVM will not perform good here as it has a very large dataset 

Q7) You are given a regression model but the problem statement requires you to perform classification. How will you do it?
Ans.-> Main objective of any regression model is to build a relationship between input variables and the target variables. Linear regression and logistic regression can be used to perform classification. Logistic regression is always better as it predicts the probability and helps in classifying it under different labels. This is done by setting a threshold frequency for each output label. If the probability of a certain variable is greater than that threshold frequency then it will be classified under that classification label.

Q8) Can we use Convolutional Neural Networks on temporal data? If so, how?
Ans.->Yes, we can use CNNs on Temporal data. Temporal data represents a state in time used for forecasting based applications. 
Precisely, in the process of conversion temporal data models to CNNs, 2 stages will be required. First stage will be the Transformation stage (for modelling Data) where identity mapping, Smoothing and Down-sampling will be done. Next Stage will be Convolution Stage where Convolution, pooling, concatenation and then Full Connection(to add Flattening Layer) will take place.

Q9) Suppose you have 5 layer CNN with each layer having a kernel of size 5*5 with zero paddings and stride 1. You pass an input of dimension 224 x 224 x 3 through this layer. There is a flattening layer after the last convolution layer. What is the dimension of output from the flattened layer?
Ans.-> Output volume Size= (224+ 2*0-5)/1 +1 =220
So, 220*220*5
No. of Parameters: each filter has 5*5*3 +1 =76 parameters   (+1 for bias)
76*5 =380

Q10) Compare the optimization capacities of SGD and Adam optimizers. Explain the working.
Ans.-> The optimization algorithm (or optimizer) is the main approach used today for training a machine learning model to minimize its error rate. There are two metrics to determine the efficacy of an optimizer: speed of convergence (the process of reaching a global optimum for gradient descent); and generalization (the model’s performance on new data).
SGD is a variant of gradient descent. Instead of performing computations on the whole dataset — which is redundant and inefficient — SGD only computes on a small subset or random selection of data examples. SGD produces the same performance as regular gradient descent when the learning rate is low.
Adam is an algorithm for gradient-based optimization of stochastic objective functions. It combines the advantages of two SGD extensions — Root Mean Square Propagation (RMSProp) and Adaptive Gradient Algorithm (AdaGrad) — and computes individual adaptive learning rates for different parameters. The Adam optimizer had the best accuracy of 99.2% in enhancing the CNN ability in classification and segmentation.

# Page 4

All three are in the link : https://github.com/Sam-Hack-King/Databyte_inductions/blob/main/Inductions_Task1.ipynb

# Page 5
 
Q1) Consider a list in python x = [1, 2, 3, 4]. What does x = x[ : : -1] do?
Ans-> It reverses the list. At first, it makes a copy of the list and then reverses the list without sorting.

Q2) When should you use a linked list instead of an array?
Ans-> When we are not sure about the number of elements required then we use a linked list as it has dynamic memory. Also, when there is a need to insert data elements in the middle then linked lists will provide better time complexity.

Q3) What will be the output for the following python code?
Ans-> {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

Q4) What will be the output for the following python code? Give an explanation.
Ans-> Output will be “Bye World”. __init__ is a constructor which initializes the attributes which invokes automatically when the object is called. 
 When the object “test” is created, it takes the “self” argument by default. The latest init method will be called and so, it will print “Bye World”.

Q5) 
a) In your choice of language(Preferably python/R), write a program that prints the numbers ranging from one to 100. But for multiples of three, print "Fizz" instead of the number, and for the multiples of five, print "Buzz." For numbers which are multiples of both three and five, print "FizzBuzz". Upload your code to GitHub repo.
Ans-> for fizzbuzz in range(1,101):
    if fizzbuzz % 3 == 0 and fizzbuzz % 5 == 0:
        print("fizzbuzz")
        continue
    elif fizzbuzz % 3 == 0:
        print("fizz")
        continue
    elif fizzbuzz % 5 == 0:
        print("buzz")
        continue
    print(fizzbuzz)
b) Consider the dataset showing the country names and their 2 digit codes. Store the entire dataset from the JSON/CSV format to a Python/R dictionary(key-value pair). Once the dictionary is made, accept the user input as follows: Country code 1, Country code 2. Display the list of country names lying in between the 2 country codes. For example, if the codes given as input are IN and US, the output should display all the country names in full that lie in between India and the United States in alphabetical order excluding these two. Outline the approach and upload your code to the GitHub repo.
c) In your choice of language(Preferably python/R), Implement a class with the name MPNeuron. This class implements an MP Neuron. Implement the following member functions in your class: (i) Initialise the constructor with the default number of inputs n = 3 and all inputs being [1,1,1] and the weights being [1,1,1] and threshold as 2.5, (ii) MP_Neuron_Input() - Accepts the number of inputs n, list of n inputs and the list of n weights with -1 being inhibitory and +1 being excitatory, and the threshold (theta) also, (iii) MP_Neuron_Evaluate() which outputs the final binary output 0 or 1 if the final value computed by the neuron is greater than the threshold. Test the MP neuron with 3 inputs and choose an appropriate combination of weights and biases to implement a 3 input NAND gate.

Ans->  Answers to A and B are in https://github.com/Sam-Hack-King/Databyte_inductions/blob/main/Inductions_Task1.ipynb below the Comment "#page 5"

Answer to C is in https://github.com/Sam-Hack-King/Databyte_inductions/blob/main/Answer%205C

#DATA SPLITING
library(AppliedPredictiveModeling)
data(twoClassData)

str(predictors)   #the twoClassData predictors are stored here
str(classes)      #the classes contains the outcome

library(caret)

set.seed(1)

#we want to create a random sample of the classes 
#this just returns the rows that it selects
trainingRows <- createDataPartition(classes, p = 0.80, list = F)
head(trainingRows)

#now we want the Training data
trainPredictors <- predictors[trainingRows,]   #now we can get the training set of the predictors
trainClasses <- classes[trainingRows]

#and the testing data, to test are training data
testPredictors <- predictors[-trainingRows,]
testClasses <- classes[-trainingRows]

str(trainPredictors)
str(trainClasses)
str(testPredictors)
str(testClasses)

#the maxDissim function from the caret package is the Maximum Dissimilarity Sampling
#this creates a sub-sample that maximizes the dissimilarity between new samples and the existing subset.
#returns the row numbers, or names that specify the sub set selected
start <- sample(1:dim(trainPredictors)[1], 15)
base <- trainPredictors[start,]
pool <- trainPredictors[-start,]
#so maxDissim needs a base and a pool of data to retrieve the data from,  here we see that we want at least 20
#elements from the pool selected,  that are obviously maximized.  Note you can add a function of your own design
#to control the maximization,   add this to the obj parameter,  this function measures dissimilarity
maxDissim(base, pool, n=20)


#RESAMPLING
set.seed(1)
repeatedSplits <- createDataPartition(trainClasses, p = 0.80, times = 3)     #so this time it will create 3 partitions of the data
str(repeatedSplits)

#just to try out a different folding technique we use the createFolds function
foldsClasses <- createFolds(trainClasses, k = 11, list = T, returnTrain = F)
#another technique that I can use is the createResample()
createResampleClasses <- createResample(trainClasses, times = 15, list = T)
#another is the createMultiFolds function which in this case create 16 folds, 15 times   so 16*15 lists.
#they are folds 
multiFoldsClasses <- createMultiFolds(trainClasses, k= 16, times = 15)
#just to clarify the createFolds() creates a list of about 10% of the data,  its a random folding of it
#the createResample() creates a full length data set that is a bootstrap sampling (meaning that it contains replacements)
#the createMultiFolds() the rows are the folds, and the columns are the cross validations

set.seed(1)
cvSplits <- createFolds(trainClasses, k = 10, returnTrain = T)
#creates 10 folds of the data
str(cvSplits)
fold1 <- cvSplits[[1]]   #gets the first fold of the Splits   this is 90% of the data
cvPredictors1 <- trainPredictors[fold1,]
cvClasses1 <- trainClasses[fold1]
nrow(trainPredictors)
nrow(cvPredictors1)


#BASIC MODEL BUILDING IN R
#the left hand side denotes the outcome and the right hand side describes how the predictors are used.
#they are seperated by the tilde ~
#example of the formula interface      modelFunction(price ~ numBedrooms + numBaths + acres, data = housingData)
#the non-formula interface specifies the predictors for the model using a matrix or a data.frame 
#example     modelFunction(x = housePredictors,  y = price)

trainPredictors <- as.matrix(trainPredictors)
knnFit <- knn3(x = trainPredictors, y = trainClasses, k = 5)
knnFit
#the knnFit is ready to predict now
testPredictions <- predict(knnFit, newdata = testPredictors, type = "class")
head(testPredictions)
str(testPredictions)

#DETERMINATION OF TUNING PARAMETERS
#the e1071 contains the tune function   errorest in the ipred     and the train function in the caret package

#SVM model is characterized by what type of kernel function the model uses. A linear relationship,  given by a linerar kernel function
# between the predictors and outcomes.   A Radial Basis Function kenrel has an additional tuning parameter associated with it
#this will impact the smoothness of the decision boundary.
library(caret)
data(GermanCredit)

GermanCredit <- GermanCredit[, -nearZeroVar(GermanCredit)]
GermanCredit$CheckingAccountStatus.lt.0 <- NULL
GermanCredit$SavingsAccountBonds.lt.100 <- NULL
GermanCredit$EmploymentDuration.lt.1 <- NULL
GermanCredit$EmploymentDuration.Unemployed <- NULL
GermanCredit$Personal.Male.Married.Widowed <- NULL
GermanCredit$Property.Unknown <- NULL
GermanCredit$Housing.ForFree <- NULL

## Split the data into training (80%) and test sets (20%)
set.seed(100)
inTrain <- createDataPartition(GermanCredit$Class, p = .8)[[1]]
GermanCreditTrain <- GermanCredit[ inTrain, ]
GermanCreditTest  <- GermanCredit[-inTrain, ]

set.seed(1056)
#lets perform a training model of svmRadial
svmFit <- train(Class ~., data = GermanCreditTrain, method = "svmRadial", preProc = c("center", "scale"), tuneLength = 10, 
                trControl = trainControl(method = "repeatedcv", repeats = 5, classProbs = T))
#first we would override several of the default values.  Lets pre-process the predictor data by centering and scaling
#you can also specify the cost values for the algorithm.   this is inserted into the tuneLength
#we can add Repeated 10 fold-cross validation in the trControl,   NOTE  need to precify that the classProbs = T
# so that we can use predict type = "probs"
svmFit
plot(svmFit, scales = list(x = list(log = 2)))
predictedClasses <- predict(svmFit, GermanCreditTest)
str(predictedClasses)
#because of the classProbs in your train method we can use type = "prob" in our predict()
predictedProbs <- predict(svmFit, newdata = GermanCreditTest, type = "prob")
head(predictedProbs)

#TODO   what does the predict Probs mean,   what does the accuracy mean,  what does the Kappa mean????

#BETWEEEN-MODEL COMPARISONS
#using the train functio with the glm   generalized linear models
set.seed(1056)
logisticReg <- train(Class~., data = GermanCreditTrain, method = "glm", trControl= trainControl(method = "repeatedcv", repeats = 5))
logisticReg
#to compare models on their cross-validation statistics,   the resamples function can be used with models 
#that share a common set of resmapled data sets.  Because of the random number seed they should have paried 
#accuracy measurements
resamp <- resamples(list(SVM = svmFit, Logistic = logisticReg))
summary(resamp)
#the NA's column is used to identify when the sample has failed
#use the diff method to asses possible differences between the models
#NOTE it is taking in the resamp
modelDifferences <- diff(resamp)
summary(modelDifferences)
#if the p values of the difference are large,  then the models fail to show any difference in performance...

#EXERCISES
#4.1 Consider the music genre data set described in Sect. 1.4. The objective
#for these data is to use the predictors to classify music samples into the
#appropriate music genre.

#A. What data splitting method would you use for these data...

#I do not have section 1.4  on the PDF for the online version,  so I will read it off the internet
#from the solutions
#When determining the data splitting method use should focus on 2 primary characteristics
#1. The number of samples relative to the number of predictors in the data
#2. The distribution of samples across classes.
#Because the number of predictors is much smaller than the number of samples,  we can split the data
#We must review the class distribution of the response
#The data shows that there is a imbalance in the classes. Because there are so many samples in the set
#Resamples would have a great chance at selecting samples across all the classes.

#when selecting a resampling method,  one needs to consider the computational time.
#The createDataPartition  function in the caret package can be used to partition the data.

#B. Using tools describe in this chapter, provide code for implementing your approach(es)
#set.seed(31)
#tenFoldCV <- createDataPartition(trainClasses, k = 10, returnTrain=T)


#4.2  Consider the permeability data set described in Sect. 1.4. The objective
#for these data is to use the predictors to model compounds’ permeability.

#A. What data splitting method(s) would you use for these data? Explain.
#because the number of samples is number smaller than the number of predictors,  splitting the data
#is probably a bad idea.  We should use reampling performance measures to select optimal tuning 
#parameters and predictive performance
#if we look at the distribution of the response,  you will see that the data is skewed.
#Because of this,  randomly selecting values could give us very bad represented training sets


#B.  Using tools described in this chapter, provide code for implementing your approach(es).
#Use Stratification to create a more representative but still random data set
#repeatedCV <- createMiltiFolds(permeability, k = 10, times = 25)



#4.3   Partial least squares (Sect. 6.3) was used to model the yield of a chemical
#manufacturing process (Sect. 1.4). The data can be found in the AppliedPre-
#dictiveModeling package and can be loaded using

library(AppliedPredictiveModeling)
data(ChemicalManufacturingProcess)

#The objective of this analysis is to find the number of PLS components
#that yields the optimal R value (Sect. 5.1). PLS models with 1 through 10
#components were each evaluated using five repeats of 10-fold cross-validation
#and the results are presented in the following table:

#A. Using the “one-standard error” method, what number of PLS components
#provides the most parsimonious model?
library(caret)
set.seed(77734)
#using ?ChemicalManufacturingProcess  we find in the Value that the Yield is the Outcome of the data.
#so now we use the train method from the caret package to use the method of "PLS" and preProcess centering and scaling
#and a resampling of repeatedcv  at 5 times
plsProfileChemMod <- train(Yield ~ ., data = ChemicalManufacturingProcess, method = "pls", preProc = c("center", "scale")
                           , tuneLength = 10, trControl = trainControl(method = "repeatedcv", repeats = 5))
 
#the plsProfilecHEMod contains the information that comes from the training set.
#if we look at the results object of it we can find a lot of information about the error and 
#for what model from the 10 given the entire object
#so here we are getting the R^2 and R^2 Standard Deviation
R2Values <- plsProfileChemMod$results[, c("ncomp", "Rsquared", "RsquaredSD")]

#SEM is the standard Error mean which is the mean standard Deviation.  SD / sqrt(N)
R2Values$RsquaredSEM <- R2Values$RsquaredSD / sqrt(length(plsProfileChemMod$control$index))
#oneSE variable  where we have geom_linerange() ...   
library(ggplot2)
oneSE <- ggplot(R2Values, aes(ncomp, Rsquared, ymin = Rsquared - RsquaredSEM, ymax = Rsquared))
oneSE + geom_linerange() + geom_pointrange() + theme_bw()

#the subset function if you remember looks for the optimal subset, so this should return the best 
#model given all the information,  in the function we specify that it will pick the ncomp
#that maximizes the R^2 value
#again R^2 is the percentage of how well the predictions line up with the actual values, aggreagated. So a 1 is perfect
#while a 0 is nothing at all.
#NOTE: Some things on R^2  it connat determine whether the coefficient estimates and predictions are biased, so
#if looking at the biase look at the residuals, mostly the plots of the residuals.
bestR2 <- subset(R2Values, ncomp == which.max(R2Values$Rsquared))
bestR2$lb <- bestR2$Rsquared - bestR2$RsquaredSEM

candR2 <- subset(R2Values, Rsquared >= bestR2$lb & ncomp < bestR2$ncomp)

#what Kuhn is talking about in this chapter again
bestR2 <- subset(R2Values, ncomp == which.max(R2Values$Rsquared))
R2Values$tolerance <- (R2Values$Rsquared - bestR2$Rsquared) / bestR2$Rsquared * 100
qplot(ncomp, tolerance, data = R2Values)
#so here we see what model works the best,  given that we started off with 10 models from the plsProfileChemMod
#we can find the best suited R^2 model from the subset() that we used,  or just looking at the data...
#plot a qplot to see how the other models work against it,  the tolerance is the measurement
#that tells us how far we are from the actual thing


#B. Compute the tolerance values for this example. If a 10 % loss in R2 is
#acceptable, then what is the optimal number of PLS components?



#C. Several other models (discussed in Part II) with varying degrees of com-plexity 
#were trained and tuned and the results are presented in Fig. 4.13. If the goal is 
#to select the model that optimizes R2, then which model(s) would you choose, and why?




#D. Prediction time, as well as model complexity (Sect. 4.8) are other factors
#to consider when selecting the optimal model(s). Given each model’s pre-
#  diction time, model complexity, and R2 estimates, which model(s) would
#you choose, and why?




#4.4 Brodnjak-Vonina et al. (2005) develop a methodology for food laborato-ries 
#to determine the type of oil from a sample. In their procedure, they used
#a gas chromatograph (an instrument that separate chemicals in a sample) to
#measure seven different fatty acids in an oil. These measurements would then
#be used to predict the type of oil in a food samples. To create their model,
#they used 96 samples2 of seven types of oils.
#These data can be found in the caret package using data(oil) . The oil
#types are contained in a factor variable called oilType. The types are pumpkin
#(coded as A ), sunflower (B), peanut (C), olive (D), soybean (E ), rapeseed (F)
#and corn (G). In R,
#data(oil)
#str(oilType)
#Factor w/ 7 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
#table(oilType)
#oilType
#A  B  C D  E  F  G
#37 26 3 7 11 10 2

#A.  Use the sample function in base R to create a completely random sample
#of 60 oils. How closely do the frequencies of the random sample match
#the original samples? Repeat this procedure several times of understand
#the variation in the sampling process.

library(caret)
data(oil)
str(oilType)


table(oilType)

#D One method for understanding the uncertainty of a test set is to use a
#confidence interval. To obtain a confidence interval for the overall accu-racy, 
#the based R function binom.test can be used. It requires the user
#to input the number of samples and the number correctly classified to
#calculate the interval. For example, suppose a test set sample of 20 oil
#samples was set aside and 76 were used for model training. For this test
#set size and a model that is about 80 % accurate (16 out of 20 correct),
#the confidence interval would be computed using
binom.test(16, 20)


sampNum <- floor(length(oilType) * 0.6) + 1
set.seed(629)
oilSplits <- vector(mode = "list", length = 20)
for(i in seq(along = oilSplits)) oilSplits[[i]] <- table(sample(oilType, size = sampNum))
head(oilSplits, 3)
#this just row binds all the list together
oilSplits <- do.call("rbind", oilSplits)
head(oilSplits, 3)

summary(oilSplits/sampNum)

set.seed(629)

oilSplits2 <- createDataPartition(oilType, p = 0.60, times = 20)
#again the lappy function sends x as the index of the oilSplits2
#because the createDataPartition creates a partition referencing the indexes of the oilSplits
oilSplits2 <- lapply(oilSplits2, function(x,y) table(y[x]), y = oilType)
head(oilSplits2, 3)

oilSplits2 <- do.call("rbind", oilSplits2)
summary(oilSplits2/sampNum)

#the following code looks at the confidence interval between 10 and 30 samples from the data.
#this is just a region of hypothesis.
getWidth <- function(values) {
  binom.test(x = floor(values["size"] * values["accuracy"]) + 1, n = values["size"])$conf.int
}
#this will create a data.frame of size, accuracy
ciInfo <- expand.grid(size = 10:30, accuracy = seq(0.7, 0.95, by = 0.01))
ciWidths <- t(apply(ciInfo, 1, getWidth))
head(ciWidths)

ciInfo$length <- ciWidths[,2] - ciWidths[,1]
levelplot(length ~ size * accuracy, data = ciInfo)






#B.  Use the caret package function createDataPartition to create a stratified
#random sample. How does this compare to the completely random sam-ples?



#C  With such a small samples size, what are the options for determining performance of the model?
#Should a test set be used?




#D One method for understanding the uncertainty of a test set is to use a
#confidence interval. To obtain a confidence interval for the overall accu-racy, 
#the based R function binom.test can be used. It requires the user
#to input the number of samples and the number correctly classified to
#calculate the interval. For example, suppose a test set sample of 20 oil
#samples was set aside and 76 were used for model training. For this test
#set size and a model that is about 80 % accurate (16 out of 20 correct),
#the confidence interval would be computed using

#> binom.test(16, 20)
#Exact binomial test data: 16 and 20
#number of successes = 16, number of trials = 20, p-value = 0.01182
#alternative hypothesis: true probability of success is not equal to 0.5 
#95 percent confidence interval:
#  0.563386 0.942666
#sample estimates:
#  probability of success
#0.8

#In this case, the width of the 95 % confidence interval is 37.9 %. Try
#different samples sizes and accuracy rates to understand the trade-off
#between the uncertainty in the results, the model performance, and the
#test set size.











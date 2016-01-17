library(AppliedPredictiveModeling)
data(segmentationOriginal)
segData <- subset(segmentationOriginal, Case=="Train")
cellID <- segData$Cell
class <- segData$Class
case <- segData$Case
#Now we are going to remove the columns
segData <- segData[, -(1:3)]
statusColNum <- grep("Status", names(segData))
segData <- segData[,-statusColNum]

library(e1071)
skewness(segData$AngleCh1)
#use the apply function to apply skewness to all the columns
#applying a skewness will either log, square root, or inverse the values
# this is done so that the distribution between the predictors is not large
skewValues <- apply(segData, 2, skewness)

#the BoxCoxTrans can apply a transformation and apply them to the new data
library(caret)
Ch1AreaTrans <- BoxCoxTrans(segData$AreaCh1)
predict(Ch1AreaTrans, head(segData$AreaCh1))

#preProcess applies this transformation to a set of predictors.
#We want to center and scale the data before we transform it
#remember that centering the data is to subtract the arerage predictor value
#to scale is to divide by the standard deviation of the predictors
pcaObject <- prcomp(segData, center = TRUE, scale. = TRUE)
percentVariance <- pcaObject$sd^2/sum(pcaObject$sd^2) * 100

#the transformed values are stroed in a sub object called x  pcaObject$x
#the sub object rotation stores the variable loadings  where row is the predictor and column is the components

#the spatialSign function can be used to make a spatial Sign

#to impute missing values, the impute package has a function impute.knn  that uses K
#nearest neighbors to estimate the missing data.  Imputation of data is to give it a value
#that seems to fit,  so that the predictor is not missing a value

#the preProcess function can be used to center scale and impute data, apply spartial, and feature extraction
trans <- preProcess(segData, method = c("BoxCox", "center", "scale", "pca"))
transformed <- predict(trans, segData)
# these transformed values are differnet because they were transformed prior to PCA
#  show head(transformed[, 1:5])

#the function nearZeroVar will return the column number of any predictors that fulfill the conditions 
#for near - zero predictors.
nearZeroVar(segData)

#the cor function can be used to filter on correlations
correlations <- cor(segData)
#show dim(correlations)
#show correlations[1:4, 1:4]
#use the corrplot
library(corrplot)
corrplot(correlations, order="hclust")  #this gives us that really nice plot  HeHe

#the findCorrelation function can be used to return column numbers denoting the predictors that are
#recommended for deletion

highCorr <- findCorrelation(correlations, cutoff = 0.75)
length(highCorr)
filteredSegData <- segData[, -highCorr]

#Kuhn says the the splits in a tree-based model are more interpretable when the dummy variables encode all the
#information for that predictor. He recommends using the full set of dummy variables when working with tree
#based models

# to make dummy variables use the dummyVars function,  this will expand the predictor  just like a dummy
#variable should be expanded.  
#dummyVars(~Mileage + Type,data = carSubset,levelsOnly = TRUE)   this expandes the Type predictor
#a dummyVariable is one in which we have a type predictor, and a mileage predictor
#using the dummyVars we will factor type into its unique factors, and create a new predictor for those factors
#then cross it with Mileage,  so now we get a group of columns by type and a indicator that relates to the Mileage
#so I think for this one we have coupe,  sedan,  muscle   as types,  they become predictors and a 1 indicates that
#they have the mileage, and a 0 indicates that they did not go that mileage.
#use the [p1]:[p2] to provide a joint effect between the two predictors
#dummyVars(~Mileage + Type + Mileage:Type,data = carSubset,levelsOnly = TRUE)
#now mileage and type will be joined together to make dummy variables, in addition to the
#dummy variables that we have for Type

#Exercises
#3.1 
library(mlbench)
data(Glass)
str(Glass)

#A.Using visualizations, explore the predictor variables to understand their
#distributions as well as the relationships between predictors.
apply(Glass[,-10],2,hist)#will display a hist for all the columns in the object
#the hist can tell us what predictors are skewed, this is indicated by the right and left
#unequalities in the data
apply(Glass[,-10],2,summary)# using this we can see the Min Max Mean Median and other things
#this will help us see the statistics of the predictors

#from the information we can see that Mg is slightly skewed, and K, Ba,Fe  are all heavy skewed

correlations <- cor(Glass[,-10])
corrplot(correlations, order="hclust")
#this will allow us to visually see the correlations between the predictors

#B.Do there appear to be any outliers in the data? Are any predictors skewed?
#so from the previous question K, BA, Fe are skewed a lot
plot(Glass$K)
plot(Glass$Ba)
plot(Glass$Fe)    # Fe is not a major outlier,  while the others are

apply(Glass[,-10], 2, skewness)
#pretty much the higher the value from the apply function here the more the skewness

#Are there any relevant transformations of one or more predictors 
#that might improve the classification model?

#so we could scale some of the data, like in Fe
#center some of the data
#PCA the Ca and RI together



#3.2

data(Soybean)

#A. Investigate the frequency distributions for the categorical predictors. Are
#any of the distributions degenerate in the ways discussed earlier in this
#chapter?

metledSoybean <- melt(Soybean, id.vars="Class")
#skipped to the Solutions to find out what they did here
#it helped cause I did not know what I was doing  =(

#B. Roughly 18 % of the data are missing. Are there particular predictors that
#are more likely to be missing? Is the pattern of missing data related to
#the classes?



#C. Develop a strategy for handling missing data, either by eliminating
#predictors or imputation.



#3.3
#A load the data

library(caret)
data(BloodBrain)

#lets find the ditributions  since everything is a int and a num we can use the densityPlot
#nope I looked at the solution  =<(


#Now reviewing what I did with the solution that is online at https://github.com/topepo/APM_Exercises/blob/master/Ch_03.pdf
library(reshape2)
#they used the library reshape2   using melt method
meltedGlass <- melt(Glass, id.vars = "Type")
#the melt function will make a melted data frame,  the id.vars identifies the id of the data.frame
#if no id or measure.vars is supplied it will assume that all are measured, and that predictors
#of factor and character types are id variables.  you can remove na values  by na.rm parameter

#they use the lattice densityplot instead of the histograms,  its a better visual

library(lattice)
densityplot(~value|variable, data = meltedGlass, scales = list(x = list(relation = "free"),
            y = list(relation = "free")), adjust = 1.25, pch="|", xlab="Predictor")

#ok the value and varibale are in meltedGlass

#from these density plots we can see.
#you can use splom to draw conditional scatter plot metrices and parallel Coordinate PLots
splom(~meltedGlass, pch=16, col=rgb(.2,.2,.2,.4), cex = .7)


#Since some of the predictors here have zero values we might want to consider using the Yeo-Johnson family of transformations
#The BoxCox  transformation of preProcess transforms the response variable while estimating transformations of predictor data.
#  the BoxCox is fast and efficient
#The YeoJohnson accommodates predictors with zero and or negative values, BoxCox has to be positive somewhat.
# center will subtract the mean of the predictor's data from the predictor values
#scale divides by the standard deviation
#zv identifies numeric predictor columns with a single value, and excludes them from further calculations.
#nzv does the same by applying the newarZeroVar function and excludes the near zero-variance predictors
#conditionalX examines the distribution of each preditor conditional on the outcome, If there is only one unique value
#   within any class then the predictor is removed 

#the BoxCox does not always work,  it does not check for the normality of the set.  It only checks for the smallest standard deviation
#so it minimizes the standard deviation with in the set of data.


yjTrans <- preProcess(Glass[,-10], method = "YeoJohnson")
yjData <- predict(yjTrans, newdata = Glass[,-10])
melted <-melt(yjData)

centerScale <- preProcess(Glass[,-10], method= c("center", "scale"))
ssData <- predict(centerScale, newdata= Glass[,-10])
splom(~ssData, pch=16, col=rgb(.2,.2,.2,.4), cex = .7)


#Exercise 2 Solutions
#the problem that I have with Exercise 2 is that the information is all Factors and there are no numbers
#the solution says that we can use the recode function of the car package to help out
#Some of the predictors in the Soybean information data.frame are integers, but classified as a factor

Soybean2 <- Soybean
table(Soybean2$temp,useNA="always")
library(car)
Soybean2$temp <- recode(Soybean2$temp, "0='low'; 1='norm'; 2 = 'high'; NA = 'missing'",
                        levels = c("low","norm","high", "missing"))
table(Soybean2$temp)

#he is going to recode months, temp, percipitation as well
#he uses the table function, to factor out how many of each group are in the predictor
#Example: $temp only has 4 types, 0  1  2  and   NA
table(Soybean2$date, useNA = "always")
Soybean2$date <- recode(Soybean2$date, "0 = 'apr'; 1 = 'may'; 2 = 'june'; 3 = 'july'; 4 = 'aug'; 5 = 'sept'; 6 = 'oct'; NA = 'missing'",
                        levels = c("apr", "may", "june","july","aug","sept","oct","missing"))
table(Soybean2$date)
table(Soybean2$precip, useNA = "always")
Soybean2$precip <- recode(Soybean2$precip, "0 = 'low'; 1 ='norm'; 2 = 'high'; NA = 'missing'",
                          levels = c("low","norm","high","missing"))
#starting with the date we can see that the distribution is not equal
table(Soybean2$date)  #july aug sept take up the most,   apr takes the least

#such as  date -  temp    for all dates, so the aggregate distribution for all temps that fall under that date

#we will be using the mosaic from vcd package and barchart from lattice package
library(vcd)
mosaic(~date + temp, data = Soybean2)

barchart(table(Soybean2$date, Soybean2$temp), auto.key = list(columns = 4, title = "temperature"))
#yeah I was right  =>)

#the following code shows what classes have missing values, it looks like it is onsided mostly
table(Soybean$Class, complete.cases(Soybean))

#the any function here takes the aggregated values of is.na(x)  (which is the predictor from the data.frame)
#and if any value is true it will return true,  but if all are false then it will return FALSE
hasMissing <- unlist(lapply(Soybean, function(x) any(is.na(x))))
hasMissing <- names(hasMissing)[hasMissing]
head(hasMissing)

#lets find the percentage of missing values for each predictor by class
#Ok so we have to understand what this apply function is doing,   it is taking the Soybean and getting 
#rid of the predictors not in hasMissing,  then it is appling the function and passing each of the Predictors
#wth the y variable   the Soybean$Class to the function
#it is making a category of the Classes with the table function, this will give the  number of Trues and FALSEs
# count the number of values with the second apply and then we have the average for that value
byPredByClass <- apply(Soybean[, hasMissing], 2, 
                       function (x, y) {
                         tab <- table(is.na(x), y)
                         tab[2,]/apply(tab, 2, sum)
                       },
                       y = Soybean$Class)

byPredByClass <- byPredByClass[apply(byPredByClass, 1, sum) > 0, ]   #get the rows that contain do not contain a 0
#this will give us only the rows that have a value in them
byPredByClass <- byPredByClass[, apply(byPredByClass, 2 ,sum) >0] #now get rid of all the columns that have zero

t(byPredByClass)#transpose it,  landscape it so we can see a better table


#looks for the factored lists that have order to them
#a factor is a categorical information of data
#a ordered factor is one that contains information like     low < med < high < extreme
#where each category in the factor has a conditional value with the others
orderedVars <- unlist(lapply(Soybean, is.ordered))
orderedVars <- names(orderedVars)[orderedVars]   #get all the ordered var names
#the orderedVars allows you to see classifications that do not have conditional values
#this is useful becasue you might want to classify something, and wonder how that something
#gets classified.
completeClasses <- as.character(unique(Soybean$Class[complete.cases(Soybean)]))  #complete.cases returns true for those that miss no values
#in this case the complete.cases returns the T or F values for each row in the data.frame
#the unieque function will group the classes from Soybean and return a unique list
#the as.character function will return the list into a list of character strings
Soybean3 <- subset(Soybean, Class %in% completeClasses)  #returns the subset of all the rows that contain the classes in completeClass
for (i in orderedVars) Soybean3[,i] <- factor(as.character(Soybean3[,i]))

#factor all the orderedColumns in the Soybean3 data.frame

dummyInfo <- dummyVars(Class ~ ., data = Soybean3)
dummies <- predict(dummyInfo, Soybean3)  #this creates 99 predictors

predDistInfo <- nearZeroVar(dummies, saveMetrics = TRUE)
head(predDistInfo)

sum(predDistInfo$nzv)   #so now we know what dummy variables are non-zero variance, and might want to get rid of them
mean(predDistInfo$nzv)  #now we get the percentage of predictor dummies that are nzv
#exspanding the dummy variabels and finding the near zero variances of those dummy variables is a good way 
#to find values that neccessary do not have a affect on the outcome of the predictions
#this is great stuff


#3.3
ncol(bbbDescr)
#remember that a nearZeroVar is a very small distribution that deviates from the norm.
#Mostly everything is the same but probably a 2-3% are different
predictInfo <- nearZeroVar(bbbDescr, saveMetrics = TRUE)
head(predictInfo)
#the nzv is created by the nearZeroVar function
rownames(predictInfo)[predictInfo$nzv]

#lets choose a nzv predictor and see what we can do with it
table(bbbDescr$a_acid)

table(bbbDescr$alert)
#lets get rid of the nzv predictors
filter1 <- bbbDescr[, !predictInfo$nzv]
ncol(filter1)

#lets look at the density plots of 127 predictors...
set.seed(532)
sampled1 <- filter1[, sample(1:ncol(filter1),8)]  #getting a smaller set?
names(sampled1)  #no  it looks like we are choosing a selct few of the predictors

library(e1071)
skew <- apply(filter1, 2, skewness)
summary(skew)
yjBBB <- preProcess(filter1, method= "YeoJohnson")
transformed <- predict(yjBBB, newdata = filter1)
sampled2 <- transformed[, names(sampled1)]   #this will get rid of those predictors that are nzv
#now filter1 is transformed in  transformed variable   and the sampled2 contains the transformed of sampled1

rawCorr <- cor(filter1)
transCorr <- cor(transformed)

ssData <- spatialSign(scale(filter1))
ssCor <- cor(ssData)

library(corrplot)
corrplot(rawCorr, order = "hclust", addgrid.col = NA, tl.pos = "n")
corrplot(transCorr, order = "hclust", addgrid.col = NA, tl.pos="n")
ssData <- spatialSign(scale(filter1))
ssCorr <- cor(ssData)
corrplot(ssCorr, order = "hclust", addgrid.col = NA, tl.pos = "n")

corrInfo <- function(x) summary(x[upper.tri(x)])
corrInfo(rawCorr)

corrInfo(transCorr)
corrInfo(ssCorr)

#rather than transform the data to resolve between predictors correlations, it may be a better idea to remove predictors
#findCorrelation is used here

thresholds <- seq(.25, .95, by = 0.05)
size <- meanCorr <- rep(NA, length(thresholds)) #create some vairables
removals <- vector(mode = "list", length = length(thresholds))

for(i in seq_along(thresholds)) {
  removals[[i]] <- findCorrelation(rawCorr, thresholds[i])  #find the correlations with the threshold
  subMat <- rawCorr[-removals[[i]], -removals[[i]]]
  size[i] <- ncol(rawCorr) -length(removals[[i]])
  meanCorr[i] <- mean(abs(subMat[upper.tri(subMat)]))    #what does the upper triangle matrix do here?
  
}

#this creates a data frame with vale threshold and what
#each of these columns have two values,  for the splitting between the size and meanCorr  in the value column
corrData <- data.frame(value = c(size, meanCorr), 
                       threshold = c(thresholds, thresholds),
                       what = rep(c("Predictors",
                                    "Average Absolute Correlation"),
                                  each = length(thresholds)))

library(subselect)
ncol(rawCorr)

#take the ratio of the smallest and largest eigenvalues, if they are greater than tolval then
#get rid of the rows given to those eigenvalues   continues to do this until condition does not apply
trimmed <- trim.matrix(rawCorr, tolval= 1000*.Machine$double.eps)$trimmedmat
ncol(trimmed)

set.seed(702)
#anneal will search for a optimal subset within the trimmed matrix,  the subset is given certain
#settings within the function,  these will be used to clarify the optimal search
sa <- anneal(trimmed, kmin = 18, kmax = 18, niter = 1000)
saMat <- rawCorr[sa$bestsets[1,], sa$bestsets[1,]]

set.seed(702)

#the genetic algorithm does the same thing, but uses a genetic evelution algorithm
#usual generation of children and facing them off against eachother to find the 
#optimal surviver
ga <- genetic(trimmed, kmin = 18, kmax = 18, nger = 1000)
gaMat <- rawCorr[ga$bestsets[1,], ga$bestsets[1,]]
fcMat <- rawCorr[-removals[size == 18][[1]],
                 -removals[size == 18][[1]]]
corrInfo(fcMat)
corrInfo(saMat)
corrInfo(gaMat)
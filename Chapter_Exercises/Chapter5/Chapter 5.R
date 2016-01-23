observed <- c(0.22, 0.83, -0.12, 0.89, -0.23, -1.30, -0.15, -1.4,
              0.62, 0.99, -0.28, 0.32, 0.34, -0.30, 0.04, -0.87,
              0.55, 1-.30, -1.15, 0.20)
predicted <- c(0.24,  0.78, -0.66, 0.53, 0.70, -0.75, -0.41, -0.43,
               0.49, 0.79, -1.19, 0.06, 0.75, -0.07, 0.43, -0.42,
               -0.25, -0.64, -1.26, -0.07)
#rediduals
residualValues <- observed - predicted
summary(residualValues)

#this is used to find a good range for the plot
axisRange <- extendrange(c(observed, predicted))
plot(observed, predicted, ylim = axisRange, xlim = axisRange)
abline(0, 1, col = "darkgrey", lty = 2)
#what we want is that there is a straight line between observed and predicted

#in this plot we want them to be on the 0 axis
plot(predicted, residualValues, ylab = "residual")
abline(h = 0, col = "darkgrey", lty = 2)

library(caret)
R2(predicted, observed)
RMSE(predicted, observed)

#lets talk about correlation,   correlation is the relationship that 2 predictors (in R) have with each other.
#to get the correlation we need to understand the Covariance between the 2 preditors
#covariance is the measure of how much the variables change together,  it is positive if variable A increases as varaible B increases
#it is negative if variable A increases and variables B decreases.  You can think of these increases and descreases on a linear regression line.
#Correlation is useually the covariance divided by the multiplicity of the standard deviations of the two data sets.
#Covariance is calculated by the Expected((X- Expected(X))(Y - Expected(Y)))

cor(predicted, observed)
cor(predicted, observed, method = "spearman")
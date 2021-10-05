#Computer Price Data -- Linear Regression

ComputerData <- read.csv(choose.files())
ComputerData

#Data description
#The business meaning of each column in the data is as below

#price: The Price of the computer
# speed: The speed
# hd: How much hard drive is present
# ram: How much ram is present in the computer
# screen: The screen size
# cd: Whether CD player is present or not
# multi: Are there multiple ports or not
# premium: If the computer premium quality
# ads: The ads value of the computer
# trend: The trend value of the computer

#Removing duplicates columns
nrow(ComputerData)
#6259

library(dplyr)
ComputerData<-distinct(ComputerData)
nrow(ComputerData)
#6183

6259-6183
#76 duplicates removed

#Defining the problem statement:
#Create a ML model which can predict the apt price of a computer
#Target Variable: price
#Predictors: RAM, HDD, CD, ports etc.

#Determining the type of Machine Learning
#Based on the problem statement you can understand that we need to create a supervised ML Regression model, as the target variable is Continuous.

library(ggplot2)
ggplot(ComputerData,aes(x=price))+geom_histogram()
#The data distribution of the target variable is satisfactory to proceed further. There are sufficient number of rows for each type of values to learn from.


#Basic Data Exploration

# Removing useless columns from the data
#There are no qualitative columns in this data

#Checking for continuous and categorical variables

length(unique(ComputerData$speed)) #6 category
length(unique(ComputerData$hd)) #continuous
length(unique(ComputerData$ram)) #6 category
length(unique(ComputerData$screen)) #3 category
length(unique(ComputerData$cd)) #2 category
length(unique(ComputerData$multi)) #2 category
length(unique(ComputerData$premium)) #2 category
length(unique(ComputerData$ads)) #continuous
length(unique(ComputerData$trend)) #continuous


# Visual Exploratory Data Analysis
# Categorical variables: Bar plot
# Continuous variables: Histogram

#Categorical Predictors: ram,screen,cd,multi,speed

library(ggplot2)
ggplot(ComputerData,aes(x=ram))+geom_bar()
ggplot(ComputerData,aes(x=screen))+geom_bar()
ggplot(ComputerData,aes(x=cd))+geom_bar()
ggplot(ComputerData,aes(x=multi))+geom_bar()
ggplot(ComputerData,aes(x=speed))+geom_bar()


#Continuous Predictors: hd,ads,trend
ggplot(ComputerData,aes(x=hd))+geom_histogram()
ggplot(ComputerData,aes(x=ads))+geom_histogram()
ggplot(ComputerData,aes(x=trend))+geom_histogram()


#All the variables are selected for further process. We will confirm in Feature Selection.

#Missing Values Treatment
colSums(is.na(ComputerData))

#No missing data


#Outlier Treatment
boxplot(ComputerData$ads)
boxplot(ComputerData$trend)
boxplot(ComputerData$hd) #positive outlier found
summary(ComputerData$hd)
Q1 <- 214
Q3 <- 528
IQR <- Q3-Q1
IQR
pos_outlier <- Q3+1.5*IQR
pos_outlier
ComputerData$hd <- ifelse(ComputerData$hd>999,998,ComputerData$hd)
boxplot(ComputerData$hd)

#Feature Selection

#Continuous vs Continuous -- Scatter plot and Correlation analysis

ggplot(ComputerData,aes(x=hd,y=price))+geom_point()+geom_smooth(method=lm)
ggplot(ComputerData,aes(x=ads,y=price))+geom_point()+geom_smooth(method = lm)
ggplot(ComputerData,aes(x=trend,y=price))+geom_point()+geom_smooth(method = lm)


cor(ComputerData$price,ComputerData$hd)
cor(ComputerData$price,ComputerData$ads)
cor(ComputerData$price,ComputerData$trend)

#We can drop ads and trend due to low correlation value 

colnames(ComputerData)
ComputerData <- ComputerData[,-c(9,10)]


#Continuous vs Categorical -- BoxPlot and ANOVA test

boxplot(ComputerData$price~factor(ComputerData$speed))
aov_speed <- aov(ComputerData$price~factor(ComputerData$speed))
summary(aov_speed)
#speed is significant

boxplot(ComputerData$price~factor(ComputerData$ram))
aov_ram <- aov(ComputerData$price~factor(ComputerData$ram))
summary(aov_ram)
#ram is significant

boxplot(ComputerData$price~factor(ComputerData$screen))
aov_screen <- aov(ComputerData$price~factor(ComputerData$screen))
summary(aov_screen)
#screen is significant

boxplot(ComputerData$price~factor(ComputerData$cd))
aov_cd <- aov(ComputerData$price~factor(ComputerData$cd)) 
summary(aov_cd)
#cd is significant

boxplot(ComputerData$price~factor(ComputerData$multi))
aov_multi <- aov(ComputerData$price~factor(ComputerData$multi))
summary(aov_multi)
#p-value=0.191>0.05 #not significant 

boxplot(ComputerData$price~factor(ComputerData$premium))
aov_prem <- aov(ComputerData$price~factor(ComputerData$premium))
summary(aov_prem)
#significant


#Removing multi column
colnames(ComputerData)
ComputerData <- ComputerData[,-7]

#Splitting the model for training and testing
library(caTools)
set.seed(100)
DataForML <- sample.split(Y=ComputerData$price,SplitRatio = 0.7)
table(DataForML)

train <- subset(ComputerData,DataForML==T) 
test <- subset(ComputerData,DataForML==F)
nrow(train)
nrow(test)

###############################################################################################
RegModel = lm(price~.,train)
RegModel
summary(RegModel)

# Measuring Goodness of Fit using R2 value on TRAINING DATA
Orig=train$price
Pred=predict(RegModel, train)
R2= 1 - (sum((Orig-Pred)^2)/sum((Orig-mean(Orig))^2))
print(paste('R2 Value is:',round(R2,2)))

# Predictions of model on Testing data
test$Prediction=predict(RegModel, test)
head(test)

# Calculating the Absolute Percentage Error for each prediction in TESTING DATA
LM_APE= 100 *(abs(test$Prediction - test$price)/test$price)
print(paste('### Mean Accuracy of Linear Regression Model is: ', 100 - mean(LM_APE)))
print(paste('### Median Accuracy of Linear Regression Model is: ', 100 - median(LM_APE)))

summary(RegModel)
#Multiple R-squared:  0.5021
#All the variables are significant

library(lmtest)
dwtest(RegModel)
#No heteroskedasticity


library(faraway)
vif(RegModel)
#No multicollinearity


































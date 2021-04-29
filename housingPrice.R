library(ISLR)
library(dplyr)
library(caret)
library(psych)

# Read in the data and show the structure of the data
HPK <- read.csv("kc_house_data.csv")
str(HPK)
names(HPK)
HPK$date <- as.numeric(HPK$date)

################################## Preprocessing ######################################

# Check percentage of missing data
percentmiss = function(x){sum(is.na(x))/ length(x) * 100}
apply(HPK,2, percentmiss)


summary(HPK)
# Check first-sight unseasonable record
filter(HPK,bedrooms == 0 |bedrooms > 10 |bathrooms == 0)

# Remove unseanable record
HPK <- HPK[!(HPK$bedrooms == 0 | HPK$bedrooms > 11 | HPK$bathrooms == 0),]



# Let's see the sale price distribution
hist(HPK$price, col = "orange")
hist(log(HPK$price),  col = "orange")
boxplot(HPK$price,col = "orange")
boxplot(log(HPK$price),col = "orange",main= "Log_Price Distribution")


# Count the confident interval 
error <- qt(0.975,df=length(HPK$price)-1)*sd(HPK$price)/sqrt(length(HPK$price))
mean(HPK$price)-error
mean(HPK$price)+error


# Identify zero or near zero variance columns
nzv <- nearZeroVar(HPK, saveMetrics=T)
nzv




# Set up a 2 x 3 plotting space
par(mfrow = c(2,3))  

# Create the loop.vector (all the columns)
loop.vector <- 1:21

# Plot histgrams of data by every column using loop
for (i in loop.vector) { 
  
  v <- HPK[,i]
  n <- names(HPK)[i]
  hist(v,
       main = NULL,
       xlab = paste(n),
       ylab = "frequency",
       col = "orange")
  
}



# Identify high correlated variables
HPK.cor <- cor(HPK[,-3])
highCorr <- findCorrelation(HPK.cor, cutoff=.7)
highCorr


# Set up a 1 x 1 plotting space
par(mfrow = c(1,1))  

# Show correlations among variables
library(GGally)
ggcorr(HPK,name = "corr", label = TRUE, hjust = 1, label_size = 3, angle = 0, size = 3,
       low = "grey", mid = "white", high = "orange")

library(corrplot)
corvalue <- cor(HPK)
corrplot.mixed(corvalue, tl.pos = "lt")


# Delete unrelated variables
HPK <- HPK[-c(1:2)]

# Create new variables "above_precentage","bathroom_bedroom_ratio"
HPK <- mutate(HPK, above_precentage = sqft_above / sqft_living,
              bathroom_bedroom_ratio = bathrooms / bedrooms)

# Create dummy value for "yr_renovated" 
HPK <- mutate(HPK, renovated = ifelse(yr_renovated == 0, "0", "1"))
HPK$renovated <- as.integer(HPK$renovated)

# Create new variables "house age" 
HPK <- mutate(HPK, house_age = 2015 - yr_built)

names(HPK)
# Remove variables "yr_renovated", "yr_build","sqft_above","sqft_basement"
drops<-c("bathrooms",  "view", "grade", "sqft_above", "sqft_basement", 
         "yr_built", "yr_renovated", "lat", "long", "sqft_living15", "sqft_lot15")
hpk <- HPK [,!(names(HPK) %in% drops)]
names(hpk)



library(leaps)

#best subset, look at up to 12 models
h.subset <- regsubsets(price ~ ., data=hpk, nvmax=12,really.big=T)
h.subset.summary <- summary(h.subset)

#review models and predictors included across M0 through M19
h.subset.summary



#identify possible skewed variables
skewValues<- apply(hpk, 2, skew) #applies skewness to each column

skewSE <- sqrt(6/nrow(hpk)) #Standard error of skewness

#anything over 2 SEs in skew is potentially problematic
abs(skewValues)/skewSE > 2

##Visualize data distributions of variables with high skew identified above
multi.hist(hpk[,abs(skewValues)/skewSE > 2], bcol="orange",
           dcol=c("black","black")) #income looks right skewed


# Create 69 new variables correspond to zipcode
library(fastDummies)
hpk_dummy <- dummy_cols(hpk, select_columns = "zipcode",remove_first_dummy = TRUE)

# Create log vlaue for price column
hpk_log <- mutate(hpk_dummy, log_price = log(hpk$price))

names(hpk_log)
# Remove column "zipcode","price"
hpk_log <- hpk_log[,-c(1,8)]
names(hpk_log)


# Split data into train and test group
set.seed(111)

# ues log price as y
trainIndex <- createDataPartition(hpk_log$log_price, p=0.75, list=F)
h.train <- hpk_log[trainIndex,]
h.test<- hpk_log[-trainIndex,]
ctrl <- trainControl(method = "cv", number=5)


########################Generalized Additive Model################################
library(mgcv)
library(gam)
library(akima)
set.seed(111)
plot(hpk_log$sqft_living, hpk_log$log_price)
gam.m1 = gam(log_price ~ s(sqft_living, k = 4), data = h.train)
plot(gam.m1)
gam.check(gam.m1)
gam.m2 = gam(log_price ~ s(sqft_living, k = 5), data = h.train)
gam.check(gam.m2)
plot(gam.m2)
gam.m3 = gam(log_price ~ s(sqft_living,k = 9), data = h.train)
gam.check(gam.m3)
gam.m4 = gam(log_price ~ s(sqft_living,k = 15), data = h.train)
plot(gam.m4)
gam.check(gam.m4)
anova(gam.m1,gam.m2,gam.m3,gam.m4, test = "F")

gam_pred <- predict(gam.m3, h.test)
mean((h.test$log_price-gam_pred)^2)
plot(h.test$log_price ~ h.test$sqft_living,xlab='Sqft_living',ylab="Real test value",
     col='orange',main="GAM Predict vs Real Test Value")
points(gam_pred~h.test$sqft_living,col='black')
summary(gam.m3)
plot(h.train$log_price ~ h.train$sqft_living,xlab='Sqft_living',ylab="log price")
library(Metrics)
mse(gam_pred, h.test$log_price)
################################ Linear Regression ##################################

#simple linear regression of price on sqft_living
hpk.lm <- lm(log_price ~ sqft_living, data=h.train)
hpk.lm

#lets visualize fitted line
plot(h.train$log_price ~ h.train$sqft_living, col= "orange")
abline(hpk.lm, col="black")

#review model fit, 
summary(hpk.lm)
par(mfrow = c(2, 2))
plot(hpk.lm,col="grey")

#What is SE of slope?
confint(hpk.lm) #defaults 95%, X-2*SE to X +2*SE*


ypredict <- predict (hpk.lm, h.test)
mean((h.test$log_price-ypredict)^2)


#extract values from model summary
#calculate t value, simple example extracting these values can be useful
coef(summary(hpk.lm))
coef(summary(hpk.lm))[2] #estimate of slope
coef(summary(hpk.lm))[2,2] #second coefficent std error
coef(summary(hpk.lm))[2,4] #extract p value of sqft_living

#hypothesis test H0 rejected for sqft_living

#residual standard error
summary(hpk.lm)


#review sum of squares break down
anova(hpk.lm)

#fit multiple regression model
price.multi.lm <- lm(log_price ~ ., data=h.train)
summary(price.multi.lm)
plot(price.multi.lm, col="grey")

price.predict <- predict (price.multi.lm, h.test)
head(price.predict)
NROW(price.predict)
mean((h.test$log_price-price.predict)^2)





#################################### Non-linear Regression ###################################
# plot sqft_living vs Log price
par(mfrow = c(1, 1))
plot(h.train$sqft_living,h.train$log_price,main="Sqft_Living vs. log Price of House",
     xlab="Sqft_Living", ylab="log Price of House", pch=19,col='orange')
abline(lm(h.train$log_price~h.train$sqft_living),col='black')

#build a another model ,polynomial regression
#create basic linear regression
basic.lm1<- lm(log_price ~ sqft_living, data=h.train)

# create 2 degree polynomial regression
poly.lm2 <- lm(log_price~poly(sqft_living,2), data=h.train)
#check the R square
summary(basic.lm1)
summary(poly.lm2)

#increase the power of polynomial regression
poly.lm3 <- lm(log_price~ poly(sqft_living,3), data=h.train)
poly.lm4 <- lm(log_price~ poly(sqft_living,4), data=h.train)
poly.lm5<- lm(log_price~ poly(sqft_living,5), data=h.train)
#use anova function to analysis of variance.
anova(basic.lm1,poly.lm2,poly.lm3,poly.lm4,poly.lm5)
summary(basic.lm1)$r.sq
summary(poly.lm2)$r.sq
summary(poly.lm3)$r.sq
# predict the log(price) value in testset
ypredict<-predict(poly.lm2,h.test)
# check MSE
mean((h.test$log_price-ypredict)^2)
plot(h.test$log_price ~ h.test$sqft_living,xlab='Sqft_living',ylab="Real test value",
     col='orange',main="Polynomail Predict vs Real Test Value")
points(ypredict~h.test$sqft_living,col='black')

#Lasso Regression
x1=model.matrix(h.train$log_price~.,h.train)
y1=h.train$log_price
newx=model.matrix(h.test$log_price~.,h.test)

library(glmnet)
grid=10^seq(10,-2,length=100)
lasso.mod=glmnet(x1,y1,alpha=1,lambda=grid)
plot(lasso.mod)
# choose best  lambda
cv.out=cv.glmnet(x1,y1,alpha=1)
plot(cv.out)      
bestlam=cv.out$lambda.min
bestlam
lasso.pred=predict(lasso.mod,s=bestlam,newx)
#check MES for Lasso regression
mean((lasso.pred-h.test$log_price)^2)
# plot Lasso predict value vs real value
plot(lasso.pred,h.test$log_price,xlim=c(12,17),ylim=c(12,17), 
     xlab="predict value",ylab="real value",col="grey",main="Lasso Predict vs. Real Test Value") 
abline(0,1,col='red')

# check lasso pick up coefficients
lasso.coef=predict(lasso.mod,type ='coefficients',s=bestlam)
lasso.coef
lasso.coef[lasso.coef!=0]
summary(lasso.coef)








####################################  models ##################################

ctrl <- trainControl(method = "cv", number=5)


#lets fit our first linear regression model
set.seed(111) #ALWAYS USE same SEED ACROSS trains to ensure identical cv folds
h.lm <- train(log_price ~ ., data= h.train, method = "lm", trControl=ctrl)

#Review cross-validation performance
h.lm

#we can review what variables were most important
varImp(h.lm)

#we can also review the final trained model just like if we used lm()
summary(h.lm)

#review residuals and actual observations against predictions
#should review interesting models using graphs and summary
par(mfrow=c(1,2)) 
plot(h.train$log_price ~ predict(h.lm), xlab="predict", ylab="actual", col = "orange")

plot(resid(h.lm) ~ predict(h.lm), xlab="predict", ylab="resid", col = "orange")





#lets fit a model that preprocess predictors on each resample fold based on our earlier findings
set.seed(111) #ALWAYS USE same SEED ACROSS trains to ensure identical cv folds
h.lm.pp <- train(log_price ~ ., data= h.train, 
                 preProcess=c("BoxCox", "scale", "center"), 
                 method = "lm", trControl=ctrl)
h.lm.pp
varImp(h.lm.pp)

## TRY K NEAREST NEIGHBOR
#lets fit our first linear regression model

#set values of k to search through, K 1 to 15
k.grid <- expand.grid(k=1:10)

set.seed(111) #ALWAYS USE same SEED ACROSS trains to ensure identical cv folds
h.knn <- train(log_price ~ ., data= h.train, method = "knn", 
               tuneGrid=k.grid, trControl=ctrl)
h.knn
varImp(h.knn)

#we can plot parameter performance
plot(h.knn)


#let preprocess K means performs better when standardized!
set.seed(111) #ALWAYS USE same SEED ACROSS trains to ensure identical cv folds
h.knn.pp <- train(log_price ~ ., data= h.train, method = "knn", 
                  preProcess=c("center", "scale"),
                  tuneGrid=k.grid, trControl=ctrl)
h.knn.pp
plot(h.knn.pp)


#lcv on forward
set.seed(111) #SEED
h.tfwd <- train(log_price ~ ., data= h.train, method = "leapForward", tuneLength=79, trControl=ctrl)
h.tfwd
getTrainPerf(h.tfwd)

#lcv on forward
set.seed(111) #SEED
h.tbwd <- train(log_price ~ ., data= h.train, method = "leapBackward", tuneLength=79, trControl=ctrl)
h.tbwd
getTrainPerf(h.tbwd)

#lasso an ridge regression
library(elasticnet)

set.seed(111) #SEED
h.ridge <- train(log_price ~ ., 
                 preProcess=c("scale"),
                 data= h.train, method = "ridge", tuneLength=10, trControl=ctrl)
h.ridge
plot(h.ridge)


set.seed(111) #SEED
h.lasso <- train(log_price ~ ., 
                 data= h.train, 
                 method = "lasso", tuneLength=100, trControl=ctrl)

set.seed(111) #SEED
h.gam <- train(log_price ~ ., 
               data= h.train, 
               method = "gam", tuneLength=100, trControl=ctrl)

##COMPARE MODEL PERFORMANCE
#let's compare all of the model resampling performance
#first lets put all trained models in a list object
h.models<- list("LM Base"=h.lm, "LM PreProc" = h.lm.pp,
               "KNN Base"=h.knn,"KNN PreProc" = h.knn.pp,
               "FORWARD" = h.tfwd, "BACKWARD" = h.tbwd,
               "RIDGE" = h.ridge, "LASSO" = h.lasso, "GAM" = h.gam)


credit.resamples<- resamples(h.models)
summary(credit.resamples)

#plot performances
bwplot(credit.resamples, metric="RMSE")
bwplot(credit.resamples, metric="Rsquared")





################################### Performance ###############################

#lets gather the models
#first lets put all trained models in a list object
models<- list("Fwd"=h.tfwd, "Bwd" = h.tbwd,
              "Ridge" = h.ridge, "Lasso"=h.lasso,
              "PCR" = h.pcr,
              "pls" = h.pls)


hitter.resamples<- resamples(models)
summary(hitter.resamples)

#plot performances
bwplot(hitter.resamples, metric="RMSE")
bwplot(hitter.resamples, metric="Rsquared")





############################### Dicision Tree ######################################################



library(rpart) 
library(tree) 
library(rpart.plot)

# Create tree
h_log.tree <- rpart(log_price ~., data=h.train)
# Summarize full tree 
h_log.tree

par(mfrow = c(1,1)) 
# Plot tree
rpart.plot(h_log.tree,type =4,box.palette="orange",main= "Tree of Log_Price")

##MSE training Error of decision tree
yhat_log.tree<- predict(h_log.tree, h.test)
mean((h.test$log_price - yhat_log.tree)^2)

# Plot tree with real price
h.tree <- rpart(10^log_price ~., data=h.train)
rpart.plot(h.tree,type =4,box.palette="orange", main= "Tree of House Price")
yhat.tree<- predict(h.tree, h.test)
mean((h.test$log_price - yhat.tree)^2)
# See price range for each zipcode
boxplot(price ~ zipcode,data = HPK,col ="orange")


#################################### Mapping ############################################

names(HPK)

# Select variable "price", "sqft_living", "zipcode","lat", and "long" for mapping
pzk <- subset(HPK, select=c("price", "sqft_living", "zipcode", "lat", "long"))
names(pzk)
pzk <- mutate(pzk, PricePerSqft = pzk$price / pzk$sqft_living)
names(pzk)
# Coculate average price, latitude and longitude for each zipcode
Pzip <- as.data.frame(aggregate(pzk[, c(1,4:6)], list(pzk$zipcode), mean))
names(Pzip)
Pzip <- rename(Pzip, zip = Group.1)
names(Pzip)
summary(Pzip$price)
summary(Pzip$PricePerSqft)

library(leaflet)

# Create map with house price per zipcode
leaflet(data = Pzip) %>% addTiles() %>%
  addMarkers(~long, ~lat, popup = ~as.character(price), label = ~as.character(zip))

# Color map marks by price level
getColor <- function(Pzip) {
  sapply(Pzip$price, function(price) {
    if(price >= 1000000) {
      "red"
    } else if(price >= 600000) {
      "orange"
    } else if(price >= 350000) {
      "beige"
    } else {
      "green"
    } })
}

icons <- awesomeIcons(
  icon = "home",
  iconColor = "black",
  library = "ion",
  markerColor = getColor(Pzip)
)

Pzip[,2] <- round(Pzip[,2],0)
values <- c("<350k", "350k-550k","550k-1m", ">1m")
pal <- colorFactor(c("chartreuse4","burlywood1","orange","red"),values,ordered = TRUE)




# Show zipcode by price
leaflet(Pzip) %>% addTiles() %>%
  addAwesomeMarkers(~long, ~lat, icon=icons, 
                    popup = ~as.character(paste("$",price)), label=~as.character(zip)) %>%
  addLegend("bottomright", pal = pal, values =~values,
            title = "House Price",
            labFormat = labelFormat(prefix = "$"),
            opacity = 1)




# Create map with house price per zipcode
leaflet(data = Pzip) %>% addTiles() %>%
  addMarkers(~long, ~lat, popup = ~as.character(PricePerSqft), label = ~as.character(zip))

# Color map marks by price level
getColor <- function(Pzip) {
  sapply(Pzip$PricePerSqft, function(PricePerSqft) {
    if(PricePerSqft >= 400) {
      "red"
    } else if(PricePerSqft >= 300) {
      "orange"
    } else if(PricePerSqft >= 200) {
      "beige"
    } else {
      "green"
    } })
}

icons <- awesomeIcons(
  icon = "home",
  iconColor = "white",
  library = "ion",
  markerColor = getColor(Pzip)
)

Pzip[,5] <- round(Pzip[,5],0)
values <- c("<200", "200-300","300-400",">400")
pal <- colorFactor(c("chartreuse4","burlywood1","orange", "red"),values,ordered = TRUE)




# Show zipcode by price
leaflet(Pzip) %>% addTiles() %>%
  addAwesomeMarkers(~long, ~lat, icon=icons, 
                    popup = ~as.character(paste("$",PricePerSqft)), label=~as.character(zip)) %>%
  addLegend("bottomright", pal = pal, values =~values,
            title = "House Price per Sqft",
            labFormat = labelFormat(prefix = "$"),
            opacity =1)

########################General Additive Model##############################

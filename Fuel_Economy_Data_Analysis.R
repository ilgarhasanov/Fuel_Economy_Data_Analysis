library(tidyverse) 
library(data.table)
library(rstudioapi)
library(recipes)
library(caret)
library(skimr)
library(purrr)
library(inspectdf)
library(mice)
library(graphics)
library(Hmisc)
library(glue)
library(highcharter)
library(plotly)
library(h2o)  

#1. Add ggplot2::mpg dataset
df = ggplot2::mpg
head(df)


#2. Make data ready for analysis doing preprocessing techniques.
df[!complete.cases(df),] %>% View()

# Outliers ----
dfnum <- df %>% select_if(is.numeric)

num_vars <- dfnum %>%
  select(-year) %>%
  names()

for_vars <- c()
for (b in 1:length(num_vars)) {
  OutVals <- boxplot(dfnum[[num_vars[b]]], plot=F)$out
  if(length(OutVals)>0){
    for_vars[b] <- num_vars[b]
  }
}
for_vars <- for_vars %>% as.data.frame() %>% drop_na() %>% pull(.) %>% as.character()
for_vars %>% length()



for (o in for_vars) {
  OutVals <- boxplot(dfnum[[o]], plot=F)$out
  mean <- mean(dfnum[[o]],na.rm=T)
  
  o3 <- ifelse(OutVals>mean,OutVals,NA) %>% na.omit() %>% as.matrix() %>% t() %>% .[1,]
  o1 <- ifelse(OutVals<mean,OutVals,NA) %>% na.omit() %>% as.matrix() %>% t() %>% .[1,]
  
  val3 <- quantile(dfnum[[o]],0.75,na.rm = T) + 1.5*IQR(dfnum[[o]],na.rm = T)
  dfnum[which(dfnum[[o]] %in% o3),o] <- val3
  
  val1 <- quantile(dfnum[[o]],0.25,na.rm = T) - 1.5*IQR(dfnum[[o]],na.rm = T)
  dfnum[which(dfnum[[o]] %in% o1),o] <- val1
}


# One hot encoding
df.chr <- df %>% select_if(is.character)
df.chr <- dummyVars(" ~ .", data = df.chr) %>% predict(newdata = df.chr) %>% as.data.frame()
df <- cbind(df.chr, dfnum ) %>% select(cty, everything())

names(df) <- names(df) %>% str_replace_all(" ", "_") %>% str_replace_all("\\(", "_") %>% str_replace_all("\\)", "_")

# Multicollinearity

target <- "cty"
features <- df %>% select(-cty) %>% names()

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df)

glm %>% summary()


# VIF
while (glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[1] >= 1.5) {
  afterVIF <- glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[-1] %>% names()
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = df)
}
glm %>% faraway::vif() %>% sort(decreasing = T) %>% names() -> features 

df <- df %>% select(cty,features)

# Standardize (Normalize) ----
df %>% glimpse()

df[,-1] <- df[,-1] %>% scale() %>% as.data.frame()



#Modelling
h2o.init()

h2o_data <- df %>% as.h2o()


# Splitting the data ----
h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'cty'
features <- df %>% select(-cty) %>% names()


# Fitting h2o model ----

model <- h2o.glm(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  nfolds = 10, seed = 123,
  lambda = 0, compute_p_values = T)

model@model$coefficients_table %>%
  as.data.frame() %>%
  dplyr::select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) %>%
  .[-1,] %>%
  arrange(desc(p_value))


# Stepwise Backward Elimination ----
while(model@model$coefficients_table %>%
      as.data.frame() %>%
      dplyr::select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] > 0.05) {
  model@model$coefficients_table %>%
    as.data.frame() %>%
    dplyr::select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train %>% as.data.frame() %>% select(target,features) %>% as.h2o()
  test_h2o <- test %>% as.data.frame() %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target,
    training_frame = train,
    validation_frame = test,
    nfolds = 10, seed = 123,
    lambda = 0, compute_p_values = T)
}

model@model$coefficients_table %>%
  as.data.frame() %>%
  dplyr::select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) 


# Model diagnostics ----
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- train_h2o %>% as.data.frame() %>% glm(f, data = .)

par(mfrow=c(2,2))
glm %>% plot()

glm %>% summary()


# Predicting the Test set results ----
y_pred <- model %>% h2o.predict(newdata = test) %>% as.data.frame()
y_pred$predict




# ----------------------------- Model evaluation -----------------------------
test_set <- test %>% as.data.frame()
residuals = test_set$cty - y_pred$predict

# Calculate RMSE (Root Mean Square Error) ----
RMSE = sqrt(mean(residuals^2))

# Calculate Adjusted R2 (R Squared) ----
y_test_mean = mean(test_set$cty)

tss = sum((test_set$cty - y_test_mean)^2) #total sum of squares
rss = sum(residuals^2) #residual sum of squares

R2 = 1 - (rss/tss); R2

n <- test_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2 = 1-(1-R2)*((n-1)/(n-k-1))

tibble(RMSE = round(RMSE,1),
       R2, Adjusted_R2)



# Plotting actual & predicted ----
my_data <- cbind(predicted = y_pred$predict,
                 observed = test_set$cty) %>% 
  as.data.frame()

g <- my_data %>% 
  ggplot(aes(predicted, observed)) + 
  geom_point(color = "darkred") + 
  geom_smooth(method=lm) + 
  labs(x="Predecited Power Output", 
       y="Observed Power Output",
       title=glue('Test: Adjusted R2 = {round(enexpr(Adjusted_R2),2)}')) +
  theme(plot.title = element_text(color="darkgreen",size=16,hjust=0.5),
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12),
        axis.title.x = element_text(size=14), 
        axis.title.y = element_text(size=14))

g %>% ggplotly()


# Check overfitting ----
y_pred_train <- model %>% h2o.predict(newdata = train) %>% as.data.frame()

train_set <- train %>% as.data.frame()
residuals = train_set$cty - y_pred_train$predict

RMSE_train = sqrt(mean(residuals^2))
y_train_mean = mean(train_set$cty)

tss = sum((train_set$cty - y_train_mean)^2)
rss = sum(residuals^2)

R2_train = 1 - (rss/tss); R2_train

n <- train_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2_train = 1-(1-R2_train)*((n-1)/(n-k-1))


# Plotting actual & predicted
my_data_train <- cbind(predicted = y_pred_train$predict,
                       observed = train_set$cty) %>% 
  as.data.frame()

g_train <- my_data_train %>% 
  ggplot(aes(predicted, observed)) + 
  geom_point(color = "darkred") + 
  geom_smooth(method=lm) + 
  labs(x="Predecited Power Output", 
       y="Observed Power Output",
       title=glue('Train: Adjusted R2 = {round(enexpr(Adjusted_R2_train),2)}')) +
  theme(plot.title = element_text(color="darkgreen",size=16,hjust=0.5),
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12),
        axis.title.x = element_text(size=14), 
        axis.title.y = element_text(size=14))

g_train %>% ggplotly()



# Compare 
library(patchwork)
g_train + g

tibble(RMSE_train = round(RMSE_train,1),
       RMSE_test = round(RMSE,1),
       
       Adjusted_R2_train,
       Adjusted_R2_test = Adjusted_R2)


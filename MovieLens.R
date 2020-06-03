################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(googledrive)) install.packages("googledrive", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
# https://drive.google.com/drive/folders/1IZcBBX0OmL9wu9AdzMBFUG8GoPbGQ38D?usp=sharing

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
head(movies)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")


# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Alternate Data Source from Google Drive Local Copy
#edx <- readRDS("~/projects/movielens/edx.rds")
#temp <- readRDS("~/projects/movielens/validation.rds")

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")


# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)




# see 'edx' data in tidy format 
edx %>% tibble()

# number of unique users and movies 
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId)) 

# show sparse matrix for sampled 100 unique userId and moveiId 
users <- sample(unique(edx$userId), 100) 
rafalib::mypar() 
edx %>% filter(userId %in% users) %>%  
   select(userId, movieId, rating) %>% 
   mutate(rating = 3) %>% 
   spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>%  
   as.matrix() %>% t(.) %>% 
   image(1:100, 1:100,. , xlab="Movies", ylab="Users") 
abline(h=0:100+0.5, v=0:100+0.5, col = "grey") 
 

# distribution of movie ratings 
edx %>%  
   dplyr::count(movieId) %>%  
   ggplot(aes(n)) +  
   geom_histogram(bins = 30, color = "black") +  
   scale_x_log10() +  
   ggtitle("Movies") 

# distribution of users 
edx %>% 
   dplyr::count(userId) %>%  
   ggplot(aes(n)) +  
   geom_histogram(bins = 30, color = "black") +  
   scale_x_log10() + 
   ggtitle("Users") 


# RMSE function for vectors of ratings and their corresponding predictors 
RMSE <- function(true_ratings, predicted_ratings){ 
   sqrt(mean((true_ratings - predicted_ratings)^2)) 
 } 
 
# Model 1: predict same rating for all movies using average of all ratings 
# predict average rating of all movies 
mu_hat <- mean(edx$rating) 
mu_hat 
# calculate rmse of this naive approach  
naive_rmse <- RMSE(validation$rating, mu_hat) 
naive_rmse 
# add rmse results in a table 
rmse_results <- data.frame(method = "Single Value Mean", RMSE = naive_rmse) 
rmse_results %>% knitr::kable() 
 

# Model 2: modeling movie effect 
# estimate movie bias 'b_i' for all movies 
mu <- mean(edx$rating) 
movie_avgs <- edx %>%  
                 group_by(movieId) %>%  
                 summarize(b_i = mean(rating - mu)) 
 # plot these movie 'bias' 
 movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black")) 
 # calculate predictions considering movie effect 
 predicted_ratings <- mu + validation %>%  
                         left_join(movie_avgs, by='movieId') %>% 
                         pull(b_i) 
 # calculate rmse after modeling movie effect 
 model_1_rmse <- RMSE(predicted_ratings, validation$rating) 
 

 rmse_results <- bind_rows(rmse_results, 
        data_frame(method="Movie Effect Model",   
                   RMSE = model_1_rmse)) 
 rmse_results %>% knitr::kable() 
 

 

 # Model 3: modeling user effect in previous model 
 # plot of avg rating for users that've rated over 100 movies 
 edx %>% group_by(userId) %>%  
            summarize(b_u = mean(rating)) %>%  
            filter(n() >= 100) %>%  
            ggplot(aes(b_u)) +  
            geom_histogram(bins = 30, color = "black") 
 # estimate user bias 'b_u' for all users 
 user_avgs <- edx %>%  
                 left_join(movie_avgs, by='movieId') %>% 
                 group_by(userId) %>% 
                 summarize(b_u = mean(rating - mu - b_i)) 
 # calculate predictions considering user effects in previous model 
 predicted_ratings <- validation %>%  
                         left_join(movie_avgs, by='movieId') %>% 
                         left_join(user_avgs, by='userId') %>% 
                         mutate(pred = mu + b_i + b_u) %>% 
                         pull(pred) 
 # calculate rmse after modeling user specific effect in previous model 
 model_2_rmse <- RMSE(predicted_ratings, validation$rating) 
 rmse_results <- bind_rows(rmse_results, 
                data_frame(method="Movie + User Effects Model",   
                           RMSE = model_2_rmse)) 
 rmse_results %>% knitr::kable() 
 

 

 # Model 4: regularizing movie + user effect model from previous models 
 # choosing the penalty term lambda 
 
 # Create additional Partition from the edx test set into edx_test and edx_val to obtain lambda
 test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
 edx_test <- edx[-test_index,]
 edx_temp <- edx[test_index,]
 
 # Make sure userId and movieId in validation set are also in edx_test set
 edx_val <- edx_temp %>% 
   semi_join(edx_test, by = "movieId") %>%
   semi_join(edx_test, by = "userId")
 
 
 # Add rows removed from validation set back into edx_test set
 removed <- anti_join(edx_temp, edx_val)
 edx_test <- rbind(edx_test, removed)
 
 # 
 
 lambdas <- seq(0, 10, 0.5) 
 
 
 rmses <- sapply(lambdas, function(l){ 
   
   mu <- mean(edx_test$rating) 
   
   b_i <- edx_test %>%  
     group_by(movieId) %>% 
     summarize(b_i = sum(rating - mu)/(n()+l)) 
   
   b_u <- edx_test %>%  
     left_join(b_i, by="movieId") %>% 
     group_by(userId) %>% 
     summarize(b_u = sum(rating - b_i - mu)/(n()+l)) 
   
   predicted_ratings <-  
     edx_val %>%  
     left_join(b_i, by = "movieId") %>% 
     left_join(b_u, by = "userId") %>% 
     mutate(pred = mu + b_i + b_u) %>% 
     pull(pred) 
   
   return(RMSE(predicted_ratings, edx_val$rating)) 
 }) 
 
 
 qplot(lambdas, rmses)  
 

 # lambda that minimizes rmse 
 lambda <- lambdas[which.min(rmses)] 
 lambda 

 # Using the optimal lambda, run the model with the original edx train and validation set
 mu <- mean(edx$rating) 
 
 b_i <- edx %>%  
   group_by(movieId) %>% 
   summarize(b_i = sum(rating - mu)/(n()+lambda)) 
 
 b_u <- edx %>%  
   left_join(b_i, by="movieId") %>% 
   group_by(userId) %>% 
   summarize(b_u = sum(rating - b_i - mu)/(n()+lambda)) 
 
 predicted_ratings <-  
   validation %>%  
   left_join(b_i, by = "movieId") %>% 
   left_join(b_u, by = "userId") %>% 
   mutate(pred = mu + b_i + b_u) %>% 
   pull(pred) 
 
 RMSE_Regularized <- RMSE(predicted_ratings, validation$rating)
 
 
 # calculate rmse after regularizing movie + user effect from previous models 
 rmse_results <- bind_rows(rmse_results, 
                           data_frame(method="Regularized Movie + User Effect Model",   
                                      RMSE = RMSE_Regularized))
 rmse_results %>% knitr::kable()
 
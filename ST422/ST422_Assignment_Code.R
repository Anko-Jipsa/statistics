# Library Setup
library(zoo)
library(astsa)
library(forecast)
library(ggplot2)
library(ggfortify)
library(tseries)
library(tidyverse)
library(tsutils)
library(MTS)
library(knitr)

# Data retrieved from Git-hub repository.
df_path = "https://raw.githubusercontent.com/Anko-Jipsa/statistics/master/ST422/"

# Mean Temperature
mt_df <- read.csv(
  paste(df_path, "/MonthlyMeanTemp.csv", sep = ""),
  header = TRUE,
  row.names = "year"
)

# Sunshine
ss_df <- read.csv(
  paste(df_path, "/MonthlySunshine.csv", sep = ""),
  header = TRUE,
  row.names = "year"
)

# Rainfall
rf_df <- read.csv(
  paste(df_path, "/MonthlyRainfall.csv", sep = ""),
  header = TRUE,
  row.names = "year"
)

# Vectorisation of CSV imported dataset
vectorised_ts = function(df) {
  #' Generate a time-series vector, ignoring aggregated level data.
  #'
  #' @param df (data.frame): A data-frame that contains the time-series vector.
  #' @return A time-series vector without aggregated level data.
  
  start_year = as.numeric(row.names(df)[1])
  
  
  vector_df = as.numeric(t(df[, 1:12])) # Flattening 2d data.
  vector_df = na.omit(vector_df) # Drop null values.
  vector_ts = ts(vector_df,
                 start = c(start_year, 1),
                 frequency = 12) # Assigning time-series.
  return(vector_ts)
}

# Transforming data
mt_ts = vectorised_ts(mt_df)
ss_ts = vectorised_ts(ss_df)
rf_ts = vectorised_ts(rf_df)

####################Q1 & Q2######################
# Preliminary analysis for identification of Time-series model
prelim_analysis = function(ts_v,
                           title = NULL,
                           start_year = NULL,
                           start_month = NULL) {
  #' Preliminary analysis for Time Series.
  #' 
  #' @description
  #' Generate time-series analysis graphs:
  #' 1. Time Series plot (General visualisation)
  #' 2. Box plot (Seasonality & volatility assessment)
  #' 3. Seasonal Plot (Deeper insight on pattern)
  #' 4. Decomposition plot (Assess trend, seasonality and remainder)
  #'
  #' @param ts_v (vector): A Time Series vector.
  #' @param title (str): Title of the plot. Defaults to NULL.
  #' @param start_year (int): A starting year of analysis. Defaults to NULL.
  #' @param start_month (int): A starting month of analysis. Defaults to NULL.
  
  if (!is.null(start)) {
    ts_v = window(ts_v,
                  start = c(start_year, start_month),
                  end = c(2020, 12))
  }
  layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))
  plot.ts(ts_v,
          ylab = "Value",
          xlab = "Date",
          main = paste(title, "TS plot", sep = " "))
  boxplot(ts_v ~ cycle(ts_v),
          ylab = "Value",
          xlab = "Month",
          main = "Box Plot")
  seasonplot(ts_v, 
             year.labels = TRUE, 
             col = 1:13,
             main = "Seasonal Plot", 
             ylab= "Value")
}

# Decomposition analysis required for Time-series model identification
decomp_analysis = function(ts_v,
                           title = NULL,
                           start_year = NULL,
                           start_month = NULL) {
  #' Decomposition analysis for Time Series.
  #' 
  #' @description
  #' Generate time-series analysis graphs:
  #' 1. Decomposition plot (Assess trend, seasonality and remainder)
  #'
  #' @param ts_v (vector): A Time Series vector.
  #' @param title (str): Title of the plot. Defaults to NULL.
  #' @param start_year (int): A starting year of analysis. Defaults to NULL.
  #' @param start_month (int): A starting month of analysis. Defaults to NULL.
  
  if (!is.null(start)) {
    ts_v = window(ts_v,
                  start = c(start_year, start_month),
                  end = c(2020, 12))
  }
  
  decomp <- stl(ts_v, s.window = 'periodic')
  autoplot(decomp) +  ggtitle("Remainder")
}

# Entire dataset
prelim_analysis(ss_ts, "Sunshine")
decomp_analysis(ss_ts, "Sunshine")

# Dataset from 2000
prelim_analysis(ss_ts, "Sunshine", 2000, 1)
decomp_analysis(ss_ts, "Sunshine", 2000, 1)

# Time-series analysis required for model identification
ts_analysis = function(ts_v,
                       title = NULL,
                       start_year = NULL,
                       start_month = NULL){
  #' ACF & PACF analysis for Time Series.
  #' 
  #' @description
  #' Generate time-series analysis graphs:
  #' 1. ACF
  #' 2. PACF
  #' 3. Seasonal Plot (Deeper insight on pattern)
  #' 4. Decomposition plot (Assess trend, seasonality and remainder)
  #'
  #' @param ts_v (vector): A Time Series vector.
  #' @param title (str): Title of the plot. Defaults to NULL.
  #' @param start_year (int): A starting year of analysis. Defaults to NULL.
  #' @param start_month (int): A starting month of analysis. Defaults to NULL.
  
  
  acf2(ts_v, main=title)
  
  
  adf.test(ts_v, k=12)
}

ts_analysis(ss_ts, "Sunshine")

# Seasonally differenced time-series
d12_ss_ts = diff(ss_ts, lag = 12)
ts_analysis(d12_ss_ts)

# Grid-search method for model exploration
grid_search_sarima = function(ts_v, non_seasonal_pqd, seasonal_pqd) {
  #' Grid-search function to find the multiplicative seasonal ARIMA model with the best fit.
  #'
  #' @description
  #' Exhaustive approach of derive multiplicative seasonal ARIMA model.
  #' Iterate through each P, Q terms of non-seasonal and seasonal components. (D is fixed.)
  #'
  #' @param ts_v (vector): A Time Series vector.
  #' @param non_seasonal_pqd (vector): Initial P, Q, D elements for non-seasonal component.
  #' @param non_seasonal_pqd (vector): Initial P, Q, D elements for seasonal component.
  
  cross_non_pqd = expand.grid(seq(max(non_seasonal_pqd[1]-1, 0), non_seasonal_pqd[1] + 1),
                              non_seasonal_pqd[2],
                              seq(max(non_seasonal_pqd[3]-1, 0), non_seasonal_pqd[3] + 1))
  
  cross_pqd = expand.grid(seq(max(seasonal_pqd[1]-1, 0), seasonal_pqd[1] + 1),
                          seasonal_pqd[2],
                          seq(max(seasonal_pqd[3]-1, 0), seasonal_pqd[3] + 1))
  
  # Empty model rank table
  mod_fit = data.frame(AIC = numeric(0), 
                       BIC = numeric(0),
                       NON_SEASONAL = numeric(0),
                       SEASONAL= numeric(0))
  count = 1
  for (i in 1:dim(cross_non_pqd)[1]){
    for (j in 1:dim(cross_pqd)[1]){
      skip_to_next <- FALSE
      
      # Status update prints
      print("************************")
      print(paste("Iteration: ", count, sep = ""))
      print(paste("Non-Seasonal: ", cross_pqd[j,], sep = ""))
      print(paste("Seasonal: ",cross_non_pqd[i,], sep = "")) 
      
      # Error catcher for ARIMA fitting (In case of optimisation error)
      mod = tryCatch(Arima(ts_v,
                           order = as.numeric(cross_non_pqd[i,]),
                           seasonal = list(order = as.numeric(cross_pqd[j,]), period = 12)), 
                     error = function(e) { skip_to_next <<- TRUE})
      if(skip_to_next) { next }
      
      # Table filling
      mod_fit[count, ] = c(mod$aic, 
                           mod$bic,
                           paste(as.character(cross_non_pqd[i,]), collapse=","), 
                           paste(as.character(cross_pqd[j,]), collapse=","))
      
      count = count + 1
    }
  }
  # Reordering
  mod_fit = mod_fit %>% arrange(AIC)
  return(mod_fit)
}

# Grid-search
grid_ss = grid_search_sarima(ss_ts, 
                             c(0,0,2), 
                             c(0,1,1))
opt_non_seasonal = as.numeric(unlist(strsplit(grid_ss[1,3], "\\,"))) # Optimal non-seasonal P, D, Q parameters
opt_seasonal = as.numeric(unlist(strsplit(grid_ss[1,4], "\\,"))) # Optimal seasonal P, D, Q parameters

# Model fitting
best_guess_ss = Arima(ss_ts,
                      order=c(0,0,2),
                      seasonal=list(order=c(0,1,1), period=12))

optimal_ss = Arima(ss_ts,
                   order=opt_non_seasonal,
                   seasonal=list(order=opt_seasonal, period=12))

auto_arima_ss = auto.arima(ss_ts)

# Model comparison
best_guess_ss # Best guess
optimal_ss # Grid-search
auto_arima_ss # auto.arima

# Residual analysis
checkresiduals(best_guess_ss)
tsdiag(best_guess_ss)

# Custom ggplot function
gg_forecast = function(model_output, n_forecast, include, title){
  #' Custom function for clear ggplot of forecast object.
  #'
  #' @param model_output (obj): a time series or time series model for which forecasts are required
  #' @param n_forecast (int): Number of periods for forecasting
  #' @param include (int): Number of periods for previous actuals 
  #' @param title (str): Title of the plot.
  #'
  
  # Actual & Forecasts
  forecast_obj = forecast(model_output, n_forecast)
  actuals = tail(forecast_obj$x, include)
  predicted = forecast_obj$mean
  
  # Date
  date_window = c(as.Date(actuals), as.Date(predicted))
  
  # Combined data.frame
  output = data.frame(time = date_window,
                      actual = c(actuals, rep(NA,length(predicted))),
                      pred = c(rep(NA,length(actuals)), predicted),
                      low_95 = c(rep(NA,length(actuals)), forecast_obj$lower[,2]),
                      up_95 = c(rep(NA,length(actuals)), forecast_obj$upper[,2]),
                      low_80 = c(rep(NA,length(actuals)), forecast_obj$lower[,1]),
                      up_80 = c(rep(NA,length(actuals)), forecast_obj$upper[,1])
  )
  
  # ggplot
  p = ggplot(output, aes(x=time), na.rm=TRUE) +  geom_line(aes(y=actual, color="actual"), na.rm=TRUE)
  # Convert to scatter plots for forecast periods less than 10
  if (n_forecast<10){
    p = p + geom_point(aes(y=pred, color="pred"), na.rm=TRUE)
  } else {
    p = p + geom_line(aes(y=pred, color="pred"), na.rm=TRUE)
  }
  
  # Confidence interval 
  p = p +
    geom_ribbon(aes(ymin=low_95, ymax=up_95, fill="95% CI"),linetype=2, alpha=0.5) +
    geom_ribbon(aes(ymin=low_80, ymax=up_80, fill="80% CI"), linetype=2, alpha=0.5)
  
  # Further aesthetics
  p = p +
    scale_colour_manual("", 
                        breaks = c("pred", "actual"),
                        values = c("darkred", "steelblue")) + 
    scale_fill_manual(values=c("grey12", "grey"), name="fill") + 
    ylab("Value") + xlab("Date") + ggtitle(title)
  
  return(p)
}

# Forecasting
gg_forecast(best_guess_ss, n_forecast=2, include=12, title="2 Months forecast plot of Sunshine")
forecast(best_guess_ss, 2)
gg_forecast(best_guess_ss, n_forecast=24, include=120, title="2 years forecast plot of Sunshine")

# Description of the following codes is identical to the above case
prelim_analysis(rf_ts, "Rainfall")
decomp_analysis(rf_ts, "Rainfall")
prelim_analysis(rf_ts, "Rainfall", 2010, 1)
decomp_analysis(rf_ts, "Rainfall", 2000, 1)

# Time-series analysis
ts_analysis(rf_ts, "Rainfall")

# Seasonally differenced Time-series analysis
d12_rf_ts = diff(rf_ts, lag = 12)
ts_analysis(d12_rf_ts)

# Grid-search
grid_rf = grid_search_sarima(rf_ts, 
                             c(1,0,1), 
                             c(0,1,1))
opt_non_seasonal = as.numeric(unlist(strsplit(grid_rf[1,3], "\\,"))) # Optimal non-seasonal P, D, Q parameters
opt_seasonal = as.numeric(unlist(strsplit(grid_rf[1,4], "\\,"))) # Optimal seasonal P, D, Q parameters

# Comparison
best_guess_rf = Arima(rf_ts,
                      order=c(1,0,1),
                      seasonal=list(order=c(0,1,1), period=12))

optimal_rf = Arima(rf_ts,
                   order=opt_non_seasonal,
                   seasonal=list(order=opt_seasonal, period=12))

auto_arima_rf = auto.arima(rf_ts)

# Models
best_guess_rf # Best guess
optimal_rf # Grid-search
auto_arima_rf # auto.arima

# Residual analysis
checkresiduals(optimal_rf)
tsdiag(optimal_rf)

# Forecast
gg_forecast(optimal_rf, n_forecast=2, include=12, title="2 months forecast plot of Rainfall")
forecast(optimal_rf, 2)
gg_forecast(optimal_rf, n_forecast=24, include=120, title="2 years forecast plot of Rainfall")

# Preliminary analysis
prelim_analysis(mt_ts, "Mean Temperature")
decomp_analysis(mt_ts, "Mean Temperature")
prelim_analysis(mt_ts, "Mean Temperature", 2010, 1)

decomp_analysis(mt_ts, "Mean Temperature", 2000, 1)

# Time-series analysis
ts_analysis(mt_ts, "Mean Temperature")

# Seasonally differenced time-series analysis
d12_mt_ts = diff(mt_ts, lag = 12)
ts_analysis(d12_mt_ts)

# Grid-search
grid_mt = grid_search_sarima(mt_ts, 
                             c(0,0,2), 
                             c(0,1,1))
opt_non_seasonal = as.numeric(unlist(strsplit(grid_mt[1,3], "\\,"))) # Optimal non-seasonal P, D, Q parameters
opt_seasonal = as.numeric(unlist(strsplit(grid_mt[1,4], "\\,"))) # Optimal seasonal P, D, Q parameters

# Model fitting
best_guess_mt = Arima(mt_ts,
                      order=c(0,0,2),
                      seasonal=list(order=c(0,1,1), period=12))

optimal_mt = Arima(mt_ts,
                   order=opt_non_seasonal,
                   seasonal=list(order=opt_seasonal, period=12))

auto_arima_mt = auto.arima(mt_ts)

# Comparison
best_guess_mt # Best guess
optimal_mt # Grid-search
auto_arima_mt # auto.arima

# Residual analysis
checkresiduals(optimal_mt)
tsdiag(optimal_mt)

# Forecasting
gg_forecast(optimal_mt, n_forecast=2, include=12, title="2 months forecast plot of Mean Temperature")
forecast(optimal_mt, 2)
gg_forecast(optimal_mt, n_forecast=24, include=120, title="2 years forecast plot of Mean Temperature")

####################Q3######################
# Rolling window function
rolling_window = function(ts_v, M=210,A=120){
  #' Function to calculate the goodness of fit and 2 months forecast for given rolling window parameters.
  #' This function is a satellite function for the iteration function below.
  #' 
  #' @description
  #' For fitting the model, this function deploys auto.arima().
  #' Goodness of fit metrics are following: RMSE, MAE.
  #'
  #' @param ts_v (vector): A Time Series vector.
  #' @param M (int): A size of window for model fitting.
  #' @param A (int): A size of window for validation.
  
  # Preset
  n = length(ts_v)
  N = ceiling((n-M-A)/A)
  
  pred_val = list()
  fitted_val = list()
  for (i in 1:N){
    # Start and End window for development & validation
    dev_start = (i-1)*A+1
    dev_end = min((i-1)*A+M, n)
    
    val_start = dev_end + 1
    val_end = min(dev_end+A, n)
    
    # Model fitting
    dev_sample = ts(ts_v[dev_start:dev_end], frequency=12)
    fit = auto.arima(dev_sample)
    
    # Prediction
    actuals = ts_v[(val_start):(val_end)]
    pred_obj = forecast(fit, A)
    
    # Status 
    print("*********************************")
    print(paste("Iteration: ", i, sep=""))
    print(paste("Modelling window: ", dev_start, "~", dev_end, sep=""))
    print(paste("Validation window: ", val_start, "~", val_end, sep=""))
    
    # Validation
    if (i==1) {
      metric = accuracy(pred_obj, actuals)
    } else {
      metric = metric + accuracy(pred_obj, actuals)
    }
  }
  
  # Averaging & Aligning the validation metrics data
  metric = data.frame(metric/N)
  metric$Window = M
  metric$Sample = n
  metric = metric[2, c( "Window", "Sample", "RMSE", "MAE")]
  row.names(metric) = NULL
  
  return(metric)
}

# List of different window values (M)
window_list = c(36, 72, 108, 144, 180)

# Window selection fuction
window_selection = function(ts_v, window_list, A){
  #' Using 'rolling_window' function to find the optimal window length.
  #' 
  #' @description
  #' Split data into train (development) and test (validation), 75% and 25% accordingly.
  #'
  #' @param ts_v (vector): A Time Series vector.
  #' @param window_list (list): A list of different window value for model fitting.
  #' @param A (int): A size of window for validation.
  
  # 75% split
  split = floor(length(ts_v)*0.75)
  #Create Train Set
  train <- window(ts_v, end = time(ts_v)[split])
  #Create Test (Validation) Set 
  test <- window(ts_v, start = time(ts_v)[split+1])
  
  # Empty variables
  output_df = data.frame(Window = numeric(0),
                         Sample = numeric(0),
                         RMSE = numeric(0),
                         MAE = numeric(0))
  fitted_list = list()
  
  # Iterative approach
  for (i in seq_along(window_list)){
    print(paste("Current iteration window (TRAIN): ", window_list[i], sep=""))
    train_output = rolling_window(train, window_list[i], A)
    
    print(paste("Current iteration window (TEST): ", window_list[i], sep=""))
    tests_output = rolling_window(test, window_list[i], A)
    
    output_df[i*2-1,] = train_output # Train sample output
    output_df[i*2,] = tests_output # Test sample output
  }
  
  return(output_df)
}

# Mean Temperature, A = 120
mt_window36 = window_selection(mt_ts, window_list, A=36)

# Rainfall, A = 120
rf_window36 = window_selection(rf_ts, window_list, A=36)

# Sunshine, A = 120
ss_window36 = window_selection(ss_ts, window_list, A=36)

kable(ss_window36, caption = "Optimal window searching result for Sunshine (Validation period, A = 36)")
kable(rf_window36, caption = "Optimal window searching result for Rainfall (Validation period, A = 36)")
kable(mt_window36, caption = "Optimal window searching result for Mean Temperature (Validation period, A = 36)")

rolling_window_forecast = function(ts_v, M, n_forecast){
  #' A function to calculate "N" steps forecasts given "M" window using auto.arima()
  #' @param ts_v (vector): A Time Series vector.
  #' @param M (int): A size of window for model fitting.
  #' @param n_forecast (int): "N" steps forecast.
  
  model = auto.arima(tail(ts_v, M))
  output = forecast(model, n_forecast)
  return(output$mean)
}

roll_window_ss = rolling_window_forecast(ss_ts, 144, 2)
roll_window_rf = rolling_window_forecast(rf_ts, 72, 2)
roll_window_mt = rolling_window_forecast(mt_ts, 144, 2)
# Custom function for comparing 2 months forecast of Q2 results with the new model.
forecast_compare = function(model_output, compare_output, n_forecast, title){
  #' Custom function for clear ggplot of forecast object.
  #'
  #' @param model_output (obj): A time series or time series model for which forecasts are required
  #' @param compare_output (obj): A time series to be compared with model output forecast
  #' @param n_forecast (int): Number of periods for forecasting
  #' @param title (str): Title of the plot.
  #'
  
  # Actual & Forecasts
  forecast_obj = forecast(model_output, n_forecast)
  predicted = forecast_obj$mean
  compare_pred = compare_output[1:n_forecast] 
  
  # Confidence interval
  c_95low = forecast_obj$lower[,2]
  c_95up = forecast_obj$upper[,2]
  c_80low = forecast_obj$lower[,1]
  c_80up = forecast_obj$upper[,1]
  
  # Date
  date_window = as.Date(as.yearmon(time(predicted)))
  
  # Combined data.frame
  output = data.frame(time = date_window,
                      q2 = as.matrix(predicted),
                      challenger = as.matrix(compare_pred),
                      low_95 = c_95low,
                      up_95 = c_95up,
                      low_80 = c_80low,
                      up_80 = c_80up)
  
  # ggplot
  p = ggplot(output, aes(x=time)) +
    geom_point(aes(y=q2, color="q2")) +
    geom_point(aes(y=challenger, color="challenger"), na.rm=TRUE) +
    theme(plot.margin = margin(2, 2, 2, 2, "cm")) + scale_x_date(date_breaks = "1 month",
                                                                 date_labels = "%B")
  
  # Confidence interval 
  p = p +
    geom_ribbon(aes(ymin=low_95, ymax=up_95, fill="95% of Q2"),linetype=2, alpha=0.5) +
    geom_ribbon(aes(ymin=low_80, ymax=up_80, fill="80% of Q2"), linetype=2, alpha=0.5)
  
  # Further aesthetics
  p = p +
    scale_colour_manual("", 
                        breaks = c("q2", "challenger"),
                        values = c("darkred", "steelblue")) + 
    scale_fill_manual(values=c("grey12", "grey"), name="CI") + 
    ylab("Value") + xlab("Date") + ggtitle(title)
  
  return(p)
}

# Question 2 and Question 3 forecasts comparison plots
forecast_compare(best_guess_ss, 
                 roll_window_ss, 
                 n_forecast=2, 
                 title="Sunshine")
forecast_compare(optimal_rf, 
                 roll_window_rf, 
                 n_forecast=2, 
                 title="Rainfall")
forecast_compare(optimal_mt, 
                 roll_window_mt, 
                 n_forecast=2, 
                 title="Mean Temperature")

####################Q4######################
# Combined data.frame for Vector ARMA modelling
varma_dat = data.frame(na.omit(cbind(ss_ts,rf_ts,mt_ts)))

# sVARMA model components
seasonal_pqd = c(1,0,1)
non_seasonal_pqd = c(0,1,1)

# Grid-search
grid_search_svarma = function(ts_df, non_seasonal_pqd, seasonal_pqd, show_step=TRUE) {
  #' Grid-search function to find the multiplicative seasonal sVARMA model with the best fit.
  #' 
  #' @section !WARNING!
  #' This function is expected to consume a significant time. 
  #' Hence, the best way to validate the final result is to plug in the model parameters into sVarma model mannually.
  #'
  #' @description
  #' Exhaustive approach of derive multiplicative seasonal sVARMA model.
  #' Iterate through each P, Q terms of non-seasonal and seasonal components. (D is fixed.)
  #'
  #' @param ts_df (data.frame): A Time Series dataframe
  #' @param non_seasonal_pqd (vector): Initial P, Q, D elements for non-seasonal component.
  #' @param non_seasonal_pqd (vector): Initial P, Q, D elements for seasonal component.
  # Ignore all warning messages.
  options(warn=2)
  cross_non_pqd = expand.grid(seq(max(non_seasonal_pqd[1]-1, 0), non_seasonal_pqd[1] + 1),
                              non_seasonal_pqd[2],
                              seq(max(non_seasonal_pqd[3]-1, 0), non_seasonal_pqd[3] + 1))
  
  cross_pqd = expand.grid(seq(max(seasonal_pqd[1]-1, 0), seasonal_pqd[1] + 1),
                          seasonal_pqd[2],
                          seq(max(seasonal_pqd[3]-1, 0), seasonal_pqd[3] + 1))
  
  # Empty model rank table
  mod_fit = data.frame(AIC = numeric(0), 
                       BIC = numeric(0),
                       NON_SEASONAL = numeric(0),
                       SEASONAL= numeric(0))
  count = 1
  for (i in 1:dim(cross_non_pqd)[1]){
    for (j in 1:dim(cross_pqd)[1]){
      skip_to_next <- FALSE
      # Status update prints
      if (show_step){
        print(paste("Step", count, sep = ": "))
      }
      
      # Error catcher for ARIMA fitting (In case of optimisation error)
      mod = tryCatch(sVARMA(ts_df,
                            order = as.numeric(cross_non_pqd[i,]),
                            s = 12,
                            sorder = as.numeric(cross_pqd[j,])), 
                     error = function(e) { skip_to_next <<- TRUE})
      if(skip_to_next) { next }
      
      # Table filling
      mod_fit[count, ] = c(mod$aic, 
                           mod$bic,
                           paste(as.character(cross_non_pqd[i,]), collapse=","), 
                           paste(as.character(cross_pqd[j,]), collapse=","))
      
      count = count + 1
    }
  }
  # Reordering
  mod_fit = mod_fit %>% arrange(AIC)
  options(warn=1)
  
  return(mod_fit)
}

# NOTE: This code takes a considerable time to execute.
# Hence, in order to compare the results, it is recommended to run individual model separately for validation.
sVarma_output = grid_search_svarma(varma_dat, seasonal_pqd, non_seasonal_pqd)

# Grid-search output
kable(sVarma_output, caption = "Seasonal VARMA Iteration table")


# Model fitting
sVARMA_final = sVARMA(varma_dat, order=c(1,0,1), sorder=c(0,1,2))
sVARMA_est = sVARMApred(sVARMA_final, 0, h=36)

# Mean Temperature prediction
svarma_mt_pred = sVARMA_est$pred[,3]
svarma_mt_err = sVARMA_est$se.err[,3]
svarma_mt_ts = sVARMA_est$pred[,3]
svarma_mt_pred = ts(sVARMA_est$pred[,3], start=c(2020, 11), frequency=12)

date_window = as.yearmon(seq(as.Date(as.yearmon(time(mt_ts))[1]),
                             as.Date(as.yearmon(time(svarma_mt_pred))[length(svarma_mt_ts)]), 
                             by = "months"), format = "%Y-%M-%d")

output = data.frame(time = date_window,
                    actual = c(mt_ts, rep(NA,length(svarma_mt_ts))),
                    pred = c(rep(NA,length(mt_ts)), svarma_mt_ts))

# Visualisation
ggplot(output[1522:dim(output)[1],], aes(x=time)) + ylab("Value") +
  geom_line(aes(y=pred, color="pred"), na.rm=TRUE) +
  geom_line(aes(y=actual, color="actual"), na.rm=TRUE) +
  scale_colour_manual("", 
                      breaks = c("pred", "actual"),
                      values = c("darkred", "steelblue"))

# Question 2 and Question 4 forecasts comparison plot
forecast_compare(optimal_mt, 
                 svarma_mt_pred, 
                 n_forecast=2, 
                 title="Mean Temperature")

library(zoo)
library(astsa)
library(forecast)
df_path = "/Users/anko/Downloads"
mt_df <- read.csv(paste(df_path,"/MonthlyMeanTemp.csv", sep=""), 
                  header=TRUE, 
                  row.names="year")

ss_df <- read.csv(paste(df_path,"/MonthlySunshine.csv", sep=""), 
                  header=TRUE, 
                  row.names="year")

rf_df <- read.csv(paste(df_path,"/MonthlyRainfall.csv", sep=""), 
                  header=TRUE, 
                  row.names="year")

ts_run = function(df){
  vector_df = as.numeric(t(df[,1:12]))
  vector_df = vector_df[!is.na(vector_df)]
  vector_ts = ts(vector_df, start=c(1884, 1), end=c(2020, 10), frequency=12)
  
  ts_df = data.frame(vector_ts)
  ts.plot(vector_ts)
  hist(vector_ts)
  par(mfrow=c(2,1))
  acf(vector_ts)
  pacf(vector_ts)
  mod = auto.arima(vector_ts)
  summary(mod)
  return(mod)
}

mod = ts_run(mt_df)
plot(forecast(mod,60), include = 30)
forecast(mod,2)

alt = Arima(vector_ts, order=c(0,0,2), seasonal=list(order=c(1,1,1), period=12))
summary(alt)
plot(forecast(alt,60), include = 60)
forecast(alt,2)

x = vector_ts
n = length(vector_ts)
M = 1600
A = 4
N = ceiling((n-M)/A) + 1
for (i in 1:N){
  x.ts = ts(x[((i-1)*A+1):(min((i-1)*A+M,n))],frequency=1)
}

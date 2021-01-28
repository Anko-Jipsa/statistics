library(e1071)
library(pracma)
library(dplyr)
df = read.csv("https://raw.githubusercontent.com/Anko-Jipsa/statistics/master/ST436/EX2/%5EKS11.csv")

# Data pre-processing
df = df %>% replace(.=="null", NA)
df = na.omit(df)
df$Date = as.Date(df$Date) 
df$Adj.Close = as.numeric(df$Adj.Close)

# Log-return
df$log_ret = c(NA, diff(log(df$Adj.Close), lag=1))
df = na.omit(df)

## Stylised facts:
# (a) The returns oscillate around zero
plot(x=df$Date, y=df$log_ret, 
     type="l", main="Log Return", ylab = "Value")
mean(df$log_ret) # Almost 0.

# (b) The returns are heavy-tailed.
xseq = linspace(min(df$log_ret), max(df$log_ret), n = 100)
x<-seq(min(df$log_ret), max(df$log_ret),by=0.02)
hist(df$log_ret, prob=TRUE, freq=F,density=50, breaks=50)
lines(xseq, dnorm(xseq, mean(df$log_ret), sd(df$log_ret)), col="blue")
kurtosis(df$log_ret) #10.81811, heavy tail

# (c) The returns display a downward skew.
skewness(df$log_ret) # Negative skewness

# (d) There is a leverage effect in the returns.
y=abs(df$log_ret[2:length(df$log_ret)])
x=df$log_ret[1:length(df$log_ret)-1]
s = ksmooth(x, y)
plot(x,y)
lines(s, lwd = 2, col = 2)

# (e) The sample autocorrelation of the returns is close to zero for almost all lags.
acf(df$log_ret)

# (f) The sample autocorrelation of the squared returns is large and positive for many lags
acf((df$log_ret)^2)

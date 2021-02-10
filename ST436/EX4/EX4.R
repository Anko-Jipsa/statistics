#Q3.a
simulate_arch = function(alpha, n=100){
  #' ARCH process simulator
  #' @param alpha (vector): vector of ARCH coefficients including omega as the first element.
  #' @param n (int): sample size
  #' @return ARCH time series of size n

  q=length(alpha)-1
  total.n=n+100
  e=rnorm(total.n)
  x=double(total.n)
  sigt=x
  sigma2=alpha[1]/(1-sum(alpha[-1]))

  if(sum(alpha[-1])>1) stop("Infinite Variance")
  if(sigma2<=0) stop("Negative Variance")
  
  x[1:q]=rnorm(q,sd=sqrt(sigma2))
  
  for (i in (q+1):total.n) 
  {
    sigt[i]=sum(alpha*c(1,x[i-(1:q)]^2))
    x[i]=e[i]*sqrt(sigt[i])
  }
  return(invisible(x[(100+1):total.n]))
}

#Q3.b
set.seed(1)
# a0 = 1, a1 = 0.5
a = simulate_arch(alpha=c(1,0.5), n=100)
plot.ts(a, main="ARCH(1), a0=1.0, a1=0.5")

set.seed(1)
# a0 = 0.1, a1 = 0.1
b = simulate_arch(alpha=c(0.1, 0.1), n=100)
plot.ts(b, main="ARCH(1), a0=0.1, a1=0.1")

set.seed(1)
# a0 = 0.7, a1 = 0.86
c = simulate_arch(alpha=c(0.7, 0.86), n=100)
plot.ts(c, main="ARCH(1), a0=0.7, a1=0.86")
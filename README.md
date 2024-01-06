# stock

In volatility.py, it uses garch model to forecast weekly volatility of the log return of stock last prices.

In optimization.py, it firstly generates the covariance matrix of daily log returns between all products. Suppose we give weights to investment of all products, then our goal is to minimize the variance of our weighted log return, which can be expressed as summation of variance and covariance. The variance is estimated as the predicted weekly volatility square, and covariances are estimated from the covariance matrix. Finally, gradient descent is applied to find the optimal weight. 

by the end of every day, if price go up (positive log return), we sell, if price go down (negative log return), we do nothing. the weight is updated daily.

in datas.zip, it contains volatility data calculated by codes.

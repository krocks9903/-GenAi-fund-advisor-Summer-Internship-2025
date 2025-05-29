import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats   
fund      = "FGPMX"     
benchmark = "^GSPC"      

hist_fund = yf.download(fund, period="3y", interval="1d")["Close"]
hist_bm   = yf.download(benchmark, period="3y", interval="1d")["Close"]

r_fund = np.log(hist_fund).diff().dropna()
r_mkt  = np.log(hist_bm).diff().dropna()

df = pd.concat([r_fund, r_mkt], axis=1).dropna()
df.columns = ["fund", "market"]
TRADING_DAYS = 252        

std_dev = df["fund"].std(ddof=0) * np.sqrt(TRADING_DAYS)

sharpe  = (df["fund"].mean() * TRADING_DAYS) / std_dev

slope, intercept, r_value, p_value, std_err = stats.linregress(df["market"], df["fund"])
beta  = slope
alpha = intercept * TRADING_DAYS    
r2    = r_value**2
metrics = {
    "3y_std_dev" : round(std_dev*100, 2),    
    "3y_sharpe"  : round(sharpe, 2),
    "beta"       : round(beta, 2),
    "alpha"      : round(alpha*100, 2),       
    "r_squared"  : round(r2*100, 1),
}
clean_metrics = {k: float(v) for k, v in metrics.items()}
print(clean_metrics)
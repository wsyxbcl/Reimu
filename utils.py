import requests

import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd

eastmoney_base = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{bench_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={time_begin}&end={time_end}"
#TODO US market

class Stock():
    #TODO refined Stock class
    pass

def data_collector(code='000300', time_begin='19900101', time_end='20991231'):
    stock_url = eastmoney_base.format(market=1, bench_code=code, time_begin=time_begin, time_end=time_end)
    stock_data = pd.DataFrame(map(lambda x: x.split(','), 
                                   requests.get(stock_url).json()["data"]["klines"]))
    stock_data.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    return stock_data

def plot_kline(stock_data):
    stock_kline = stock_data.set_index("date")
    stock_kline.index = pd.to_datetime(stock_kline.index)
    stock_kline = stock_kline.astype(float)
    ma_value = (5, 20, 60)
    kwargs = dict(type='candle', mav=ma_value, volume=True, figratio=(11, 8), figscale=0.85)
    fig, axes = mpf.plot(stock_kline, **kwargs, style='yahoo', returnfig=True)
    mav_leg = axes[0].legend(['ma_{}'.format(i) for i in ma_value], loc=9, ncol=3, 
                              prop={'size': 7}, fancybox=True, borderaxespad=0.)
    mav_leg.get_frame().set_alpha(0.4)
    # axes[0].set_title("test_title")
    fig.savefig('./test.png', dpi=300)

if __name__ == '__main__':
    x = data_collector()
    plot_kline(x)
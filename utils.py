import datetime
import os
import pickle
import re
import time

import mplfinance as mpf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

matplotlib.rcParams['font.family'] = ['Source Han Sans']

data_path = './data'
_test_path = './demo'
#TODO API for split-adjusted share prices
eastmoney_base = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{bench_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={time_begin}&end={time_end}"

#TODO HK market

#TODO still need a check
# "沪市（主板、科创板、基金）、深市（主板、中小板、创业板、基金）", guided by Maple
# http://www.szse.cn/
# http://www.sse.com.cn/
_test_stock_code = ['600000', '688111', '510010', '000001', '002001', '300001', '159001']

# For filtering eastmoney searchapi
# "TypeUS" seems to be a strong factor, but with uncertain meaning 
# MktNum: MarketName
stock_market =  {'0': "SZ", '1': "SH", '105': "US", '106': "US", '107': "US", '156': "US"} #TODO figure out US market
market_emoji =  {"SZ": '🇨🇳', "SH": '🇨🇳', "US": '🇺🇸', "HK": '🇭🇰'}
# SecurityType: SecurityTypeName
stock_type = {'1': "沪A", 
              '25': "科创板", 
              '2': "深A", 
              '8': "基金", 
              '5': "指数"}

# kline color style
mc = {'candle': {'up': '#fe3032', 'down': '#00b060'},
      'edge': {'up': '#fe3032', 'down': '#00b060'},
      'wick': {'up': '#fe3032', 'down': '#00b060'},
      'ohlc': {'up': '#fe3032', 'down': '#00b060'},
      'volume': {'up': '#fd6b6c', 'down': '#4dc790'},
      'vcedge': {'up': '#1f77b4', 'down': '#1f77b4'},
      'vcdopcod': False,
      'alpha': 0.7}

class QueryError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class Stock:
    #TODO refined Stock class
    def __init__(self, code, name, market_id, type_id):
        self.code = code
        self.name = name
        self.market_id = market_id
        self.type_id = type_id
    @property
    def stock_market(self):
        return self.stock_market[market_id]
    @property
    def stock_type(self):
        return self.stock_type[type_id]

    def __repr__(self):
        return "<Stock code={0.code!r} name={0.name!r} market_id={0.market_id!r} type_id={0.type_id!r}>".format(self)
    def __str__(self):
        return "{0.name!s}({0.code!s})".format(self)

class Stock_mix:
    """
    Mixed stocks, Inherence not considered currently 
    """
    def __init__(self, code, name, stock_list, holding_ratio, create_time=datetime.datetime.utcnow()):
        self.code = code
        self.name = name
        self.stock_list = stock_list
        self.holding_ratio = holding_ratio
        self.create_time = create_time #TODO just date is fine, consider using utf-8

    def draw(self, output=os.path.join(_test_path, 'stock_mix.jpg')):
        labels = [stock.name for stock in self.stock_list]
        ratios = [ratio for ratio in self.holding_ratio]
        colors = plt.cm.get_cmap('tab20c').colors
        fig1, ax1 = plt.subplots()
        ax1.pie(ratios, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90) #draw circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle) # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        ax1.set_title(f"{self.name}({self.code}), created at {self.create_time}(UTC)")
        plt.tight_layout()
        fig.savefig(output, dpi=300)

    def save(self):
        output=os.path.join(data_path, self.code+'.pickle')
        with open(output, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        
    def get_profit_ratio(self, mix_data, matrix_close_price, date_ref=None):
        # mix_data can be calculated from matrix_close_price, 
        # just for saving time here

        # Match datetime with utc, NOTICE that timezone are assumed to be same
        # in one mixed stock 

        # Timezone convert
        if date_ref is None:
            date_ref_index_utc = pd.Timestamp(self.create_time).tz_localize('UTC')
        else:
            date_ref_index_utc = pd.Timestamp(date_ref).tz_localize('UTC')
        date_ref_index = (
            date_ref_index_utc.tz_convert('Asia/Shanghai').date()
            if stock_market[self.stock_list[0].market_id] == 'SZ' or 'SH'
            else date_ref_index_utc.tz_convert('US/Eastern').date())
        get_value = lambda x: (x.index.values[0], x.values[0])
        mix_price_ref_idx, mix_price_ref = get_value(mix_data.loc[mix_data['date'] == str(date_ref_index)]['close'])
        mix_price_today_idx, _ = get_value(mix_data.loc[mix_data['date'] == list(mix_data['date'])[-1]]['close'])
        profit_ratio = (mix_data['close'].values - mix_price_ref) / mix_price_ref

        matrix_price_ref = matrix_close_price[:, mix_price_ref_idx]
        matrix_price_today = matrix_close_price[:, mix_price_today_idx]
        # matrix_profit_ratio = (matrix_close_price - matrix_price_ref.reshape(-1, 1)) / matrix_price_ref.reshape(-1, 1)
        stock_profit_ratio = (matrix_price_today - matrix_price_ref) / matrix_price_ref
        return profit_ratio, stock_profit_ratio

    def __repr__(self):
        return "<Stock_mix code={0.code!r} name={0.name!r}>".format(self)
    def __str__(self):
        return "{0.name!s}({0.code!s}), created at {0.create_time!s}\n".format(self) +\
               "\n".join("{!s}\t{:.1%}".format(stock, ratio)\
               for stock, ratio in zip(self.stock_list, self.holding_ratio))


def _query_test(stock_list):
    """
    Leave test 
    """
    for stock in stock_list:
        try:
            stock_query(stock, echo=True)
        except QueryError:
            print(f"QueryError on {stock}")

#TODO minus plot -> /compare command
#TODO real time price -> /realtime command
def stock_query(keyword, echo=False):
    """
    borrowed from https://github.com/pengnanxiaomeimei/stock_data_analysis/
    Not ideal but works.
    """
    search_keyword = keyword.split('(')[0]
    # Advanced search, to deal with duplicated query results
    # Only stock_name / stock_code is supported now
    # i.e. stock_name(stock_code or stock_code(stock_name
    try:
        advanced_search_keyword = keyword.split('(')[1]
        print(advanced_search_keyword)
    except IndexError:
        advanced_search_keyword = None

    if (local_stock := search_keyword+'.pickle') in os.listdir(data_path):
        #TODO do query instead of match
        with open(os.path.join(data_path, local_stock), 'rb') as f:
            local_stock = pickle.load(f)
        return [local_stock] # to make return value consistent

    if keyword.isspace() or not keyword:
        raise QueryError("Empty query")
    # configure search API
    query_url = 'http://searchapi.eastmoney.com/api/suggest/get'
    cb_param_pre = 'jQuery112406497239864696334_'
    token = 'D43BF722C8E33BDC906FB84D85E326E8'
    time_stamp = int(round(time.time() * 1000))
    str_parameter = '?cb=' + cb_param_pre + str(time_stamp)
    str_parameter += '&input=' + search_keyword
    str_parameter += '&token=' + token
    str_parameter += '&type=14' # for Securities entry
    str_parameter += '&count=5'
    str_parameter += '&_=' + str(time_stamp)
    query_url = query_url + str_parameter

    r = requests.get(query_url)
    p2 = re.compile(r'[(](.*)[)]', re.S)
    result = re.findall(p2, r.content.decode('utf-8'))[0]
    try:
        mes_dict = eval(result)
    except NameError as e:
        raise QueryError("Can't find keyword") from e
    query_result = mes_dict['QuotationCodeTable']['Data']

    # Filter result
    stock_list = [Stock(code=x['Code'], name=x['Name'], market_id=x['MktNum'], type_id=x['SecurityType']) 
                  for x in query_result if x['MktNum'] in stock_market and \
                                           (x['SecurityType'] in stock_type or x['Classify'] == "UsStock") and\
                                           x["SecurityTypeName"] != "曾用"]
    if advanced_search_keyword is not None:
        stock_list = [x for x in stock_list if (x.name == advanced_search_keyword) or (x.code == advanced_search_keyword)] 
        print(stock_list)
    if not stock_list:
        raise QueryError(f"Result not in A-SHARE\n{query_result}")
    if echo:
        print(stock_list)
    return stock_list

def data_collector(stock, time_begin='19900101', time_end='20991231'):
    stock_url = eastmoney_base.format(market=stock.market_id, 
                                      bench_code=stock.code, 
                                      time_begin=time_begin, 
                                      time_end=time_end)
    try:
        stock_data = pd.DataFrame(map(lambda x: x.split(','), 
                                      requests.get(stock_url).json()["data"]["klines"]))
    except TypeError as e:
        raise QueryError("Can't find kline data") from e
    stock_data.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    return stock_data

def mix_data_collector(stock_mix, time_begin='20210101', time_end='20991231', time_ref=None):
    """
    Collecting and postprocessing for Stock_mix, where only close price are collected
    Noted that long time range can cause date inconsistency
    """
    if time_ref is None:
        time_ref = time_begin
    #TODO use time_ref data when plot kline and 'value' as ylabel
    stock_data = [data_collector(stock, time_begin=time_begin, time_end=time_end) for stock in stock_mix.stock_list]
    # Checking whether the dates are consistent
    try:
        matrix_date = np.array([np.array(stock['date']) for stock in stock_data])
    except ValueError:
        # operands could not be broadcast together
        raise
    if not np.equal(matrix_date[0], matrix_date).all():
        print("date inconsistent")

    matrix_close_price = np.array([np.array(stock['close']) for stock in stock_data]).astype(float)
    # matrix_volume = np.array([np.array(stock['volume']) for stock in stock_data]).astype(float)
    # only need close price here
    close_price_ref = matrix_close_price[:, 0]
    stock_share_ratios = stock_mix.holding_ratio / close_price_ref
    value_mix = np.average(matrix_close_price, axis=0, weights=stock_share_ratios) 
    value_mix = value_mix / value_mix[0] # norm to 1
    mix_data = stock_data[0].copy()
    mix_data = mix_data.drop(['money', 'change', 'volume'], axis=1)
    mix_data['close'] = value_mix
    # mix_data['volume'] = volume_mix
    # Data redundancy, rather inelegant here, might go PR on mplfinance (or simplily using plot instead)
    mix_data['low'] = mix_data['open'] = mix_data['high'] = np.zeros(len(stock_data[0]['date']))
    return mix_data, matrix_close_price # to be used in profit analysis

def plot_kline(stock_data, title='', plot_type='candle', volume=True, macd=False, output=os.path.join(_test_path, 'kline.jpg')):
    #TODO analysis, e.g. MACD, RSI
    # issue#316 of mplfinance might be helpful
    stock_kline = stock_data.set_index("date")
    stock_kline.index = pd.to_datetime(stock_kline.index)
    stock_kline = stock_kline.astype(float)
    if plot_type == 'line':
        ma_value = ()
    else:
        ma_value = (5, 10, 20)
    if macd:
        # Using MACD(12,26,9) here
        # https://en.wikipedia.org/wiki/MACD
        ema_12 = stock_data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = stock_data['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - macd_signal

        apds = [# mpf.make_addplot(ema_12, color='lime'),
                # mpf.make_addplot(ema_26, color='c'),
                mpf.make_addplot(macd_histogram, type='bar', width=0.7, panel=1,
                                 color='dimgray', alpha=1, secondary_y=False),
                mpf.make_addplot(macd_line, panel=1, color='b', width=0.5, secondary_y=True),
                mpf.make_addplot(macd_signal, panel=1, color='r', width=0.5, secondary_y=True)]
        if volume:
            kwargs = dict(type=plot_type, addplot=apds, mav=ma_value, volume=volume, 
                          figratio=(4, 3), figscale=0.85, volume_panel=2, panel_ratios=(6, 3, 2))
        else:
            kwargs = dict(type=plot_type, addplot=apds, mav=ma_value, volume=volume, 
                          figratio=(11, 8), figscale=0.85)
    else:
        kwargs = dict(type=plot_type, mav=ma_value, volume=volume, figratio=(11, 8), figscale=0.85)
    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size':8}, marketcolors=mc)
    fig, axes = mpf.plot(stock_kline, **kwargs, 
                         style=style, 
                         scale_padding={'left': 0.4, 'top': 1, 'right': 1, 'bottom': 1}, 
                         returnfig=True)
    if ma_value:
        mav_leg = axes[0].legend(['ma_{}'.format(i) for i in ma_value], loc=9, ncol=3, 
                                prop={'size': 7}, fancybox=True, borderaxespad=0.)
        mav_leg.get_frame().set_alpha(0.4)
    if macd:
        mav_leg = axes[3].legend(["MACD", "MACD Signal"], loc=9, ncol=3, 
                                prop={'size': 7}, fancybox=True, borderaxespad=0.)
        mav_leg.get_frame().set_alpha(0.4)
    axes[0].set_title(title)
    fig.savefig(output, dpi=300)

def plot_profitline(stock_data, profit_ratio, title='', output=os.path.join(_test_path, 'profitline.jpg')):
    stock_data['close'] = profit_ratio
    stock_profitline = stock_data.set_index("date")
    stock_profitline.index = pd.to_datetime(stock_profitline.index)
    stock_profitline = stock_profitline.astype(float)
    kwargs = dict(type='line', volume=False, ylabel='Return Rate', figratio=(11, 8), figscale=0.85)
    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size':8})
    fig, axes = mpf.plot(stock_profitline, **kwargs, 
                         style=style, 
                         scale_padding={'left': 0.1, 'top': 1, 'right': 1, 'bottom': 1}, 
                         returnfig=True)
    axes[0].set_title(title)
    fig.savefig(output, dpi=300)
   
def plot_stock_profit(stock_mix, stock_profit_ratio, title='', output=os.path.join(_test_path, 'profitstocks.jpg')):
    stock_df = pd.DataFrame()
    stock_df['name'] = [stock.name for stock in stock_mix.stock_list]
    stock_df['profit'] = stock_profit_ratio
    stock_df['colors'] = ['red' if x >= 0 else 'green' for x in stock_df['profit']]
    stock_df.sort_values('profit', inplace=True)
    stock_df.reset_index(inplace=True)

    matplotlib.rcParams['font.family'] = ['Source Han Sans']
    plt.figure()
    plt.hlines(y=stock_df.index, xmin=0, xmax=stock_df.profit, color=stock_df.colors, alpha=0.4, linewidth=5)

    plt.gca().set(ylabel='Stock', xlabel='Return Rate')
    plt.yticks(stock_df.index, stock_df.name, fontsize=8)
    plt.title(title)
    # plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(output, dpi=300)

def gen_stock_mix(mix_code, mix_name, stock_names, holding_ratios):
    stock_list = []
    for stock_name in stock_names:
        query_result = stock_query(stock_name, echo=True)
        if len(query_result) == 1:
            stock_list.append(query_result[0])
        else:
            print("multiple query results on "+stock_name)
    stock_mix = Stock_mix(code=mix_code, name=mix_name, stock_list=stock_list, 
                              holding_ratio=holding_ratios)
    stock_mix.save()
    print(stock_mix)
    return stock_mix

def main():
    # kline plot test
    x = data_collector(stock_query('000300', echo=True)[0], time_begin='20210101')
    plot_kline(x, title='test_kline', plot_type='candle', volume=True, macd=True)

    # Query test
    # _query_test(_test_stock_code)

    # Stock_mix test
    # enl_stock_name = [] # a list of query keywords (#TODO check: result must be unique)
    # enl_stock_ratio = [1 / len(enl_stock_name)] * len(enl_stock_name)
    # enl_stock_mix = gen_stock_mix(mix_code='enl001', mix_name="enl stock mix", 
    #                               stock_names=enl_stock_name, holding_ratios=enl_stock_ratio)

    # Stock_mix object Loading & Pie plot, return rate analysis test

    enl_stock_mix = stock_query('enl001')[0]
    enl_stock_mix.draw()
    mix_data, matrix_close_price = mix_data_collector(enl_stock_mix)
    datetime_yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    profit_ratio, stock_profit_ratio = enl_stock_mix.get_profit_ratio(mix_data, matrix_close_price, date_ref=datetime_yesterday)
    print(profit_ratio)
    print(stock_profit_ratio)
    plot_kline(mix_data, title=enl_stock_mix.name, plot_type='line', volume=False)
    matplotlib.rcParams['font.family'] = ['Source Han Sans']
    plot_profitline(mix_data, profit_ratio)
    plot_stock_profit(enl_stock_mix, stock_profit_ratio)

if __name__ == '__main__':
    main()

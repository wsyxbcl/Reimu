import datetime
import hashlib
import io
import logging
import re
import toml

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineQuery, ParseMode, \
    InputTextMessageContent, InlineQueryResultPhoto
import matplotlib.font_manager

from utils import *

#TODO complete logging
#TODO /compare
#TODO /realtime

config = toml.load("reimu.toml")

# Configure logging
logging.basicConfig(filename="./hakurei_bot.log",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=config["telegram"]["token"])
dp = Dispatcher(bot)

def get_time_range(days_interval=100):
    """
    return ({days_interval} days ago, today + 1)
    """
    time_end = datetime.datetime.utcnow() + datetime.timedelta(days=1)
    time_begin = time_end - datetime.timedelta(days=days_interval)
    return (time_begin.strftime("%Y%m%d"), time_end.strftime("%Y%m%d"))

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message):
    await message.reply("Busy in stock marketing, no time for talk")

@dp.message_handler(commands=['kline'])
async def kline(message):
    logging.info(f'{message.chat.id}: {message.text}')
    stock_list = stock_query(keyword=message.text.split()[1])
    logging.info(f'query result:{stock_list}')
    if len(stock_list) == 1:
        stock = stock_list[0]
        try:
            time_range = get_time_range(int(message.text.split()[2]))
        except IndexError:
            time_range = get_time_range()
        buf = io.BytesIO()
        if type(stock) == Stock_mix:
            stock_data, _ = mix_data_collector(stock, price='average', time_begin=time_range[0], time_end=time_range[1])
            plot_kline(stock_data=stock_data, 
                       title=f'kline of {stock.code}',
                       plot_type='line',
                       output=buf)
        else:        
            plot_kline(stock_data=data_collector(stock, time_range[0], time_range[1]), 
                       title=f'kline of {stock.code}',
                       output=buf)
        buf.seek(0)
        await message.reply_photo(buf, caption=stock.code+' '+stock.name)
    else:
        await message.reply("Find multiple results:\n"+'\n'.join(['/kline ```'+stock.code+'```'+' '+stock.name for stock in stock_list]), 
                            parse_mode=ParseMode.MARKDOWN) 

@dp.message_handler(commands=['define'])
async def define(message):
    #TODO consider argparser or inline keyboard
    logging.info(f'{message.chat.id}: {message.text}')
    code = message.text.split()[1]
    name = message.text.split()[2]
    stock_names = message.text[message.text.find("(")+1:message.text.find(")")]
    stock_list = [stock_name for stock_name in stock_names.split()]
    if message.text.split()[-1] == 'equal':
        holding_ratio = [1 / len(stock_list)] * len(stock_list)
    stock_mix = gen_stock_mix(code, name, stock_names=stock_list, holding_ratios=holding_ratio)
    stock_mix.save()
    logging.info(f'creating stock mix:{stock_mix}')
    buf = io.BytesIO()
    stock_mix.draw(output=buf)
    buf.seek(0)
    await message.reply_photo(buf, caption=stock_mix.code+' '+stock_mix.name+" created")

@dp.message_handler(commands=['check'])
async def check(message):
    """
    check and plot the profit ratio of given stock mix
    """
    logging.info(f'{message.chat.id}: {message.text}')
    #TODO achieve argparser-like behavior here
    stock_list = stock_query(keyword=message.text.split()[1])
    logging.info(f'query result:{stock_list}')
    if len(stock_list) == 1 and type(stock_mix := stock_list[0]) is Stock_mix:
        if '-d' in message.text or '--detail' in message.text:
            time_begin = stock_mix.create_time.strftime("%Y%m%d")
        else:
            try:
                time_begin, _ = get_time_range(int(message.text.split()[2]))
            except IndexError:
                time_begin = stock_mix.create_time.strftime("%Y%m%d")
        buf = io.BytesIO()
        stock_data, matrix_close_price = mix_data_collector(stock_mix, price='average', time_begin=time_begin)
        profit_ratio, stock_profit_ratio = stock_mix.get_profit_ratio(stock_data, matrix_close_price, date_ref=stock_mix.create_time)
        if '-d' in message.text or '--detail' in message.text:
            plot_stock_profit(stock_mix, stock_profit_ratio, 
                              title=f'{stock_mix.code} {stock_mix.name} from {time_begin} (UTC)',
                              output=buf)
        else:
            plot_profitline(stock_data, profit_ratio, 
                            title=f'{stock_mix.code} {stock_mix.name} from {time_begin} (UTC)',
                            output=buf)
        buf.seek(0)
        await message.reply_photo(buf, caption=stock_mix.code+' '+stock_mix.name+\
                                               "\n当前收益率: {:.2%}".format(profit_ratio[-1]))
    else:
        pass
        #TODO if there will be stock_mix query

@dp.message_handler(commands=['now'])
async def now(message):
    logging.info(f'{message.chat.id}: {message.text}')
    #TODO achieve argparser-like behavior here
    stock_list = stock_query(keyword=message.text.split()[1])
    logging.info(f'query result:{stock_list}')
    if len(stock_list) == 1 and type(stock_mix := stock_list[0]) is Stock_mix:
        try:
            time_begin, _ = get_time_range(int(message.text.split()[2]))
        except IndexError:
            time_begin = stock_mix.create_time.strftime("%Y%m%d")
        buf = io.BytesIO()
        datetime_yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        stock_data, matrix_close_price = mix_data_collector(stock_mix, price='average', time_begin=datetime_yesterday)
        profit_ratio, stock_profit_ratio = stock_mix.get_profit_ratio(stock_data, matrix_close_price, date_ref=stock_mix.create_time)
        plot_stock_profit(stock_mix, stock_profit_ratio, 
                          title=f'{stock.code} {stock.name} from {datetime_yesterday.strftime("%Y%m%d")}(UTC)', 
                          output=buf)
        await message.reply_photo(buf, caption=stock_mix.code+' '+stock_mix.name+\
                                               "\n今日收益率: {:.2%}".format(profit_ratio[-1]))
    else:
        pass
        #TODO combined with dayline


#@dp.message_handler()
#TODO inline mode to be developed

# @dp.inline_handler()
# async def inline_echo(inline_query: types.InlineQuery):
#     text = inline_query.query or "echo"
#     results = []
# 
#     code = inline_query.query
# 
#     results.append(
#         types.InlineQueryResultPhoto(
#             id=1, photo_url=mainURL, title=text, thumb_url=mainURL
#         )
#     )
# 
#     await bot.answer_inline_query(inline_query.id, results=results, cache_time=1)

if __name__ == '__main__':
    matplotlib.rcParams['font.family'] = ['Source Han Sans']
    executor.start_polling(dp, skip_updates=True)

datetime.datetime.utcnow() - datetime.timedelta(days=1)

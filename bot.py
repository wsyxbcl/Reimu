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

def get_time_range(days_interval=120):
    """
    return ({days_interval} days ago, today + 1)
    """
    time_end = datetime.datetime.utcnow() + datetime.timedelta(days=1)
    time_begin = time_end - datetime.timedelta(days=days_interval)
    return (time_begin.strftime("%Y%m%d"), time_end.strftime("%Y%m%d"))

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message):
    await message.reply("Busy in stock trading, no time for talk") 

@dp.message_handler(commands=['kline'])
async def kline(message):
    logging.info(f'{message.chat.id}: {message.text}')
    stock_list = stock_query(keyword=message.text.split()[1])
    logging.info(f'query result:{stock_list}')
    # Get time_range from user input
    try:
        days_interval = message.text.split()[2]
    except IndexError:
        time_begin, time_end = get_time_range()
        days_interval = ''
        macd = True
    else:
        time_begin, time_end = get_time_range(int(days_interval))
        macd = (int(days_interval) >= 100)

    if len(stock_list) == 1:
        stock = stock_list[0]
        buf = io.BytesIO()
        if type(stock) == Stock_mix:
            time_mix_created = stock.create_time.strftime("%Y%m%d")
            if days_interval == '':
                time_begin = time_mix_created
                macd = False
            stock_data, _ = mix_data_collector(stock, time_begin=time_begin, time_end=time_end, 
                                               time_ref=time_mix_created)
            plot_kline(stock_data=stock_data, title=f'kline of {stock.code}',
                       plot_type='line', volume=False, macd=macd, output=buf)
        else:        
            plot_kline(stock_data=data_collector(stock, time_begin, time_end), 
                       title=f'kline of {stock.code}',
                       macd=macd, output=buf)
        buf.seek(0)
        await message.reply_photo(buf, caption=stock.code+' '+stock.name, reply_markup=types.ReplyKeyboardRemove())
    else:
        keyboard_markup = types.InlineKeyboardMarkup()
        for stock in stock_list:
            stock_emoji = market_emoji[stock_market[stock.market_id]]
            keyboard_markup.row(types.InlineKeyboardButton(' '.join([stock_emoji, stock.code, stock.name]), 
                                callback_data='/kline '+stock.code+'('+stock.name+' '+days_interval))
        # add exit button
        keyboard_markup.row(types.InlineKeyboardButton('exit', callback_data='exit'))
        await message.reply_photo("AgACAgUAAxkBAAIDymATQjsOZbFGxGqQMKt-Q_MUyUXdAAL1qjEbgr-YVC1IVvhlQFtLfoeybnQAAwEAAwIAA20AA47jAQABHgQ", 
                                  caption="Find multiple results", reply_markup=keyboard_markup)

@dp.callback_query_handler(lambda cb: '/kline' in cb.data)
@dp.callback_query_handler(text='exit') #TODO need expection on multi-user behavior?
async def inline_kline_answer_callback_handler(query):
    logging.info(f'{query.inline_message_id}: {query.data}')
    if query.data == 'exit':
        await query.message.delete()
        return 1
    stock_list = stock_query(keyword=query.data.split()[1])
    logging.info(f'query result:{stock_list}')
    stock = stock_list[0]
    try:
        days_interval = query.data.split()[2]
    except IndexError:
        time_begin, time_end = get_time_range()
        macd = True
    else:
        time_begin, time_end = get_time_range(int(days_interval))
        macd = (int(days_interval) >= 100)
    buf = io.BytesIO()
    if type(stock) == Stock_mix:
        time_mix_created = stock.create_time.strftime("%Y%m%d")
        if days_interval == '':
            time_begin = time_mix_created
            macd = False
        stock_data, _ = mix_data_collector(stock, time_begin=time_begin, 
                                           time_end=time_end, time_ref=time_mix_created)
        plot_kline(stock_data=stock_data, title=f'kline of {stock.code}',
                   plot_type='line', volume=False, macd=macd, output=buf)
    else:        
        plot_kline(stock_data=data_collector(stock, time_begin, time_end), 
                   title=f'kline of {stock.code}',
                   macd=macd, output=buf)
    buf.seek(0)
    await query.message.edit_media(types.InputMediaPhoto(media=buf, caption=stock.code+' '+stock.name))
 
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
        time_now = datetime.datetime.utcnow().strftime("%Y%m%d %H:%M:%S")
        stock_data, matrix_close_price = mix_data_collector(stock_mix, time_begin=time_begin)
        profit_ratio, stock_profit_ratio = stock_mix.get_profit_ratio(stock_data, matrix_close_price, 
                                                                      date_ref=stock_mix.create_time)
        if '-d' in message.text or '--detail' in message.text:
            plot_stock_profit(stock_mix, stock_profit_ratio, 
                              title=f'{stock_mix.name} {time_begin}-{time_now} (UTC)',
                              output=buf)
        else:
            plot_profitline(stock_data, profit_ratio, 
                            title=f'Return rate of {stock_mix.code}, {time_begin}-{time_now} (UTC)',
                            output=buf)
        buf.seek(0)
        await message.reply_photo(buf, caption=stock_mix.code+' '+stock_mix.name+\
                                               "\nCurrent return rate: {:.2%}".format(profit_ratio[-1]))
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
        time_now = datetime.datetime.utcnow().strftime("%Y%m%d %H:%M:%S")
        datetime_yesterday = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime("%Y%m%d")
        stock_data, matrix_close_price = mix_data_collector(stock_mix, time_begin=datetime_yesterday)
        profit_ratio, stock_profit_ratio = stock_mix.get_profit_ratio(stock_data, matrix_close_price, 
                                                                      date_ref=datetime_yesterday)
        plot_stock_profit(stock_mix, stock_profit_ratio, 
                          title=f'{stock_mix.name} {datetime_yesterday}-{time_now} (UTC)', 
                          output=buf)
        buf.seek(0)
        await message.reply_photo(buf, caption=stock_mix.code+' '+stock_mix.name+\
                                               "\nToday's return rate: {:.2%}".format(profit_ratio[-1]))
    else:
        pass
        #TODO combined with dayline

# To get photo file_id
# @dp.message_handler(content_types=['photo'])
# async def echo(message):
#     await message.answer(str(message.photo[0].file_id))

if __name__ == '__main__':
    matplotlib.rcParams['font.family'] = ['Source Han Sans']
    executor.start_polling(dp, skip_updates=True)

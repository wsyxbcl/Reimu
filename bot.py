import argparse
import datetime
import hashlib
import io
import logging
import re
import sys
import toml

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineQuery, ParseMode, message, \
    InputTextMessageContent, InlineQueryResultPhoto
import matplotlib.font_manager

from utils import *
from commands import argparse_kline, argparse_define, argparse_xqimport, argparse_status, argparse_now, argparse_compare
import xueqiu_portfolio

sys.setrecursionlimit(10000) # To support the usage of bs4

_market_emoji =  {"SZ": '🇨🇳', "SH": '🇨🇳', "US": '🇺🇸', "NASDAQ": '🇺🇸', "NYSE": '🇺🇸', "AMEX": '🇺🇸', "HK": '🇭🇰'}
# file id for the picture, i.e. placeholder of inline keyboard
_file_id_inline = "AgACAgUAAxkBAAIDymATQjsOZbFGxGqQMKt-Q_MUyUXdAAL1qjEbgr-YVC1IVvhlQFtLfoeybnQAAwEAAwIAA20AA47jAQABHgQ" 

config = toml.load("reimu.toml")

# Configure logging
logging.basicConfig(filename="./hakurei_bot.log",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=config["telegram"]["token"])
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def send_welcome(message):
    await message.reply("Busy in stock trading, no time for talks.\n"
                        "/help if you need, contribute or build your own from https://github.com/wsyxbcl/Reimu") 

@dp.message_handler(commands=['help'])
async def send_help(message):
    await message.reply("Current functions are: \n"
                        "/kline, /status, /now /define and /compare, "
                        "parameters are retrieved using argparse, add -h accordingly for detail.")

@dp.message_handler(commands=['kline'])
async def kline(message, query=None):
    logging.info(f'{message.text}')
    args = argparse_kline(message.text)
    if args.help:
        await message.reply(argparse_kline.__doc__)
        return 0
    try:
        stock_list = stock_query(keyword=args.keyword[0], filter_md5=args.md5)
    except IndexError:
        raise
    logging.info(f'query result:{stock_list}')
    # Get time_range from user input
    if args.days:
        time_arg = '-d '+args.days
        time_begin, time_end = get_time_range(int(args.days))
        macd = (int(args.days) >= 100)
    else:
        time_arg = None
        time_begin, time_end = get_time_range()
        macd = True

    if len(stock_list) == 1:
        stock = stock_list[0]
        buf = io.BytesIO()
        if type(stock) == Stock_mix:
            stock_info = ''
            time_mix_created = stock.create_time.strftime("%Y%m%d")
            if args.days is None:
                time_begin = time_mix_created
                macd = False
                stock_data, _ = await mix_data_collector_async(stock, time_begin=time_begin, time_end=time_end, 
                                                               time_ref='oldest')
            else:
                stock_data, _ = await mix_data_collector_async(stock, time_begin=time_begin, time_end=time_end, 
                                                               time_ref='created')
            plot_kline(stock_data=stock_data, title=f'kline of {stock.code}',
                       plot_type='line', volume=False, macd=macd, output=buf)
        else:
            if stock_info := stock.company_info:
                stock_info = f"[INFO]({stock_info})"
            plot_kline(stock_data=stock.collect_data_daily(time_begin, time_end), 
                       title=f'kline of {stock.code}', plot_type='hollow_candle', macd=macd, output=buf)
        buf.seek(0)
        kline_caption = (' '.join([stock.code, stock.name, stock_info])).replace('*', '\*') # A-share sucks!
        if args.md5:
            # Not open to user input, can only result from inline keyboard callback
            await query.message.edit_media(types.InputMediaPhoto(media=buf, caption=kline_caption, parse_mode=ParseMode.MARKDOWN))
        else:
            await message.reply_photo(buf, caption=kline_caption, parse_mode=ParseMode.MARKDOWN)
    else:
        # get user's selection from inline keyboard
        keyboard_markup = types.InlineKeyboardMarkup()
        for stock in stock_list:
            stock_emoji = _market_emoji[stock.market]
            keyboard_markup.row(types.InlineKeyboardButton(' '.join([stock_emoji, stock.code, stock.name]), 
                                callback_data=' '.join(filter(None, ['/kline', time_arg, '-e', stock.md5, args.keyword[0]]))))
        # add exit button
        keyboard_markup.row(types.InlineKeyboardButton('exit', callback_data='exit'))
        await message.reply_photo(_file_id_inline, caption="Found multiple results", reply_markup=keyboard_markup)

@dp.callback_query_handler(lambda cb: '/kline' in cb.data)
@dp.callback_query_handler(text='exit') #TODO need expection on multi-user behavior?
async def inline_kline_answer_callback_handler(query):
    logging.info(f'{query.inline_message_id}: {query.data}')
    if query.data == 'exit':
        await query.message.delete()
        return 1
    await kline(message.Message(text=query.data), query=query)

@dp.message_handler(commands=['define'])
async def define(message):
    try:
        args = argparse_define(message.text)
    except argparse.ArgumentError:
        pass 
    if args.help:
        await message.reply(argparse_define.__doc__)
        return 0
    logging.info(f'{message.chat.id}: {message.text}')
    try:
        (code, name) = args.code_and_name
    except ValueError:
        raise #Wrong command received
    stock_list = args.stock_list
    if args.weights is None:
        holding_ratio = [1 / len(stock_list)] * len(stock_list)
    else:
        holding_ratio = [float(w) for w in args.weights]
    stock_mix = gen_stock_mix(code, name, stock_names=stock_list, holding_ratios=holding_ratio, create_time=datetime.datetime.utcnow())
    logging.info(f'creating stock mix:{stock_mix}')
    if type(stock_mix) is dict:
        candidate_list = stock_mix
        await message.reply("Try using code to specify the following stocks:\n"+str(candidate_list).replace('*', '\*'), parse_mode=ParseMode.MARKDOWN)
        return 2
    buf = io.BytesIO()
    stock_mix.draw(output=buf)
    buf.seek(0)
    await message.reply_photo(buf, caption=stock_mix.code+' '+stock_mix.name+" created")

@dp.message_handler(commands=['xqimport'])
async def xqimport(message):
    try:
        args = argparse_xqimport(message.text)
    except argparse.ArgumentError:
        pass
    if args.help:
        await message.reply(argparse_xqimport.__doc__)
        return 0
    logging.info(f'{message.chat.id}: {message.text}')
    try:
        (xqcode, query_code) = args.codes
    except ValueError:
        raise #Wrong command received
    if args.preferred_name is None:
        name = xueqiu_portfolio.get_pfdata(xqcode)['name']
    else:
        name = args.preferred_name
    asset_list = xueqiu_portfolio.get_asset_list(xqcode)
    stock_list = list(asset_list['Code'])
    holding_ratio = [float(weight.strip('%')) for weight in list(asset_list['Weight'])]
    stock_mix = gen_stock_mix(query_code, name, stock_names=stock_list, holding_ratios=holding_ratio, create_time=datetime.datetime.utcnow())
    
    buf = io.BytesIO()
    stock_mix.draw(output=buf)
    buf.seek(0)
    await message.reply_photo(buf, caption=stock_mix.code+' '+stock_mix.name+" created")

@dp.message_handler(commands=['status'])
async def status(message):
    """
    plot the return rate of given stock mix
    """
    logging.info(f'{message.chat.id}: {message.text}')
    args = argparse_status(message.text)
    if args.help:
        await message.reply(argparse_status.__doc__)
        return 0
    stock_list = stock_query(keyword=args.stock_mix_code)
    logging.info(f'query result:{stock_list}')
    if len(stock_list) == 1 and type(stock_mix := stock_list[0]) is Stock_mix:
        buf = io.BytesIO()
        if args.ls:
            stock_mix.draw(output=buf)
            buf.seek(0)
            await message.reply_photo(buf, caption=str(stock_mix))
        else:
            # add buffer time for non-trading days
            time_created = stock_mix.create_time.strftime("%Y%m%d")
            time_begin = (stock_mix.create_time - datetime.timedelta(days=9)).strftime("%Y%m%d")
            time_now = datetime.datetime.utcnow().strftime("%Y%m%d %H:%M:%S")
            # stock_data, matrix_close_price = mix_data_collector(stock_mix, time_begin=time_begin)

            stock_data, matrix_close_price = await mix_data_collector_async(stock_mix, time_begin=time_begin)
            profit_ratio, stock_profit_ratio = stock_mix.get_profit_ratio(stock_data, matrix_close_price, 
                                                                          date_ref=stock_mix.create_time)
            if args.detail:
                plot_stock_profit(stock_mix, stock_profit_ratio, 
                                  title=f'{stock_mix.name} {time_created}-{time_now} (UTC)',
                                  output=buf)
            else:
                plot_profitline(stock_data, profit_ratio, 
                                title=f'Return rate of {stock_mix.code}, {time_created}-{time_now} (UTC)',
                                output=buf)
            buf.seek(0)
            await message.reply_photo(buf, caption=stock_mix.code+' '+stock_mix.name+\
                                                   "\nCurrent return rate: {:.2%}".format(profit_ratio[-1]))
    else:
        pass

@dp.message_handler(commands=['now'])
async def now(message, query=None):
    logging.info(f'{message.text}')
    try:
        args = argparse_now(message.text)
    except argparse.ArgumentError:
        pass
    if args.help:
        await message.reply(argparse_now.__doc__)
        return 0
    try:
        stock_list = stock_query(keyword=args.keyword, filter_md5=args.md5)
    except IndexError:
        raise
    logging.info(f'query result:{stock_list}')
    if len(stock_list) == 1:
        buf = io.BytesIO()
        stock = stock_list[0]
        if type(stock_mix := stock_list[0]) is Stock_mix:
            try:
                time_begin, _ = get_time_range(int(message.text.split()[2]))
            except IndexError:
                time_begin = stock_mix.create_time.strftime("%Y%m%d")
            datetime_ref = (datetime.datetime.utcnow() - datetime.timedelta(days=30)).strftime("%Y%m%d") # refer to last trading day
            stock_data, matrix_close_price = await mix_data_collector_async(stock_mix, time_begin=datetime_ref)
            profit_ratio, stock_profit_ratio = stock_mix.get_profit_ratio(stock_data, matrix_close_price, 
                                                                        date_ref='latest')
            plot_stock_profit(stock_mix, stock_profit_ratio, 
                            title=f'{stock_mix.name} Latest return rate', #TODO add a timestamp
                            output=buf)
            buf.seek(0)
            await message.reply_photo(buf, caption=stock_mix.code+' '+stock_mix.name+\
                                                "\nLatest return rate: {:.2%}".format(profit_ratio[-1]))
        else:
            if stock_info := stock.company_info:
                stock_info = f"[INFO]({stock_info})"
            plot_kline(stock_data=stock.collect_data_live(), 
                       title=f'Live price of {stock.code} (UTC+8)', plot_type='line', volume=True, macd=False, output=buf)
            buf.seek(0)
            now_caption = (' '.join([stock.code, stock.name, stock_info])).replace('*', '\*') # A-share sucks!
            if args.md5:
                # Not open to user input, can only result from inline keyboard callback
                await query.message.edit_media(types.InputMediaPhoto(media=buf, caption=now_caption, parse_mode=ParseMode.MARKDOWN))
            else:
                await message.reply_photo(buf, caption=now_caption, parse_mode=ParseMode.MARKDOWN)
    else:
        # get user's selection from inline keyboard
        keyboard_markup = types.InlineKeyboardMarkup()
        for stock in stock_list:
            stock_emoji = _market_emoji[stock.market]
            keyboard_markup.row(types.InlineKeyboardButton(' '.join([stock_emoji, stock.code, stock.name]), 
                                callback_data=' '.join(filter(None, ['/now', '-e', stock.md5, args.keyword]))))
        # add exit button
        keyboard_markup.row(types.InlineKeyboardButton('exit', callback_data='exit'))
        await message.reply_photo(_file_id_inline, caption="Find multiple results", reply_markup=keyboard_markup)

@dp.callback_query_handler(lambda cb: '/now' in cb.data)
@dp.callback_query_handler(text='exit')
async def inline_now_answer_callback_handler(query):
    logging.info(f'{query.inline_message_id}: {query.data}')
    if query.data == 'exit':
        await query.message.delete()
        return 1
    await now(message.Message(text=query.data), query=query)

@dp.message_handler(commands=['compare'])
async def compare(message):
    try:
        args = argparse_compare(message.text)
    except argparse.ArgumentError:
        pass 
    if args.help:
        await message.reply(argparse_compare.__doc__)
        return 0
    logging.info(f'{message.chat.id}: {message.text}')

    time_begin, _ = get_time_range(int(args.days))
    stock_list = [stock_query(keyword)[0] for keyword in args.stocks]
    buf = io.BytesIO()
    await plot_return_rate_anlys_async(stock_list, date_begin=time_begin, output=buf)
    buf.seek(0)
    await message.reply_photo(buf, caption='')
    
# To get photo file_id
# @dp.message_handler(content_types=['photo'])
# async def echo(message):
#     await message.answer(str(message.photo[0].file_id))

if __name__ == '__main__':
    matplotlib.rcParams['font.family'] = ['Source Han Sans']
    executor.start_polling(dp, skip_updates=True)

import argparse
import datetime
import hashlib
import io
import logging
import random
import re
import sys
import toml
import time

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.bot import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Message,
)
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

router = Router()
dp = Dispatcher()
dp.include_router(router)


@router.message(Command("start"))
async def send_welcome(message: Message):
    await message.reply("Busy in stock trading, no time for talks.\n"
                        "/help if you need, contribute or build your own from https://github.com/wsyxbcl/Reimu") 

@router.message(Command("help"))
async def send_help(message: Message):
    await message.reply("Current functions are: \n"
                        "/kline, /status, /now /define and /compare, "
                        "parameters are retrieved using argparse, add -h accordingly for detail.")

async def kline(message: Message, command_text: str | None = None, query: CallbackQuery | None = None):
    text = command_text or message.text
    logging.info(f'{text}')
    args = argparse_kline(text)
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
                try:
                    stock_data, _ = await mix_data_collector_async(stock, time_begin=time_begin, time_end=time_end, 
                                                               time_ref='oldest')
                except QueryError as e:
                    await message.reply(f"Failed to collect data: {e}")
                    return 1
            else:
                try:
                    stock_data, _ = await mix_data_collector_async(stock, time_begin=time_begin, time_end=time_end, 
                                                               time_ref='created')
                except QueryError as e:
                    await message.reply(f"Failed to collect data: {e}")
                    return 1
            plot_kline(stock_data=stock_data, title=f'kline of {stock.code}',
                       plot_type='line', volume=False, macd=macd, output=buf)
        else:
            if stock_info := stock.company_info:
                stock_info = f"[INFO]({stock_info})"
            try:
                stock_data = stock.collect_data_daily(time_begin, time_end)
            except QueryError as e:
                await message.reply(f"Failed to collect data: {e}")
                return 1
            plot_kline(stock_data=stock_data, 
                       title=f'kline of {stock.code}', plot_type='hollow_candle', macd=macd, output=buf)
        buf.seek(0)
        kline_caption = (' '.join([stock.code, stock.name, stock_info])).replace('*', r'\*') # A-share sucks!
        photo = BufferedInputFile(buf.getvalue(), filename="kline.png")
        if args.md5:
            # Not open to user input, can only result from inline keyboard callback
            await query.message.edit_media(InputMediaPhoto(media=photo, caption=kline_caption, parse_mode=ParseMode.MARKDOWN))
        else:
            await message.reply_photo(photo, caption=kline_caption, parse_mode=ParseMode.MARKDOWN)
    else:
        # get user's selection from inline keyboard
        keyboard_markup = InlineKeyboardMarkup(inline_keyboard=[])
        for stock in stock_list:
            stock_emoji = _market_emoji[stock.market]
            keyboard_markup.row(InlineKeyboardButton(' '.join([stock_emoji, stock.code, stock.name]), 
                                callback_data=' '.join(filter(None, ['/kline', time_arg, '-e', stock.md5, args.keyword[0]]))))
        # add exit button
        keyboard_markup.row(InlineKeyboardButton('exit', callback_data='exit'))
        await message.reply_photo(_file_id_inline, caption="Found multiple results", reply_markup=keyboard_markup)

@router.message(Command("kline"))
async def kline_command(message: Message):
    await kline(message)

@router.callback_query(F.data.startswith('/kline'))
async def inline_kline_answer_callback_handler(query: CallbackQuery):
    logging.info(f'{query.inline_message_id}: {query.data}')
    await kline(query.message, command_text=query.data, query=query)

@router.callback_query(F.data == 'exit') #TODO need expection on multi-user behavior?
async def exit_callback(query: CallbackQuery):
    logging.info(f'{query.inline_message_id}: exit')
    await query.message.delete()
    return 1

@router.message(Command("define"))
async def define(message: Message):
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
        await message.reply("Try using code to specify the following stocks:\n"+str(candidate_list).replace('*', r'\*'), parse_mode=ParseMode.MARKDOWN)
        return 2
    buf = io.BytesIO()
    stock_mix.draw(output=buf)
    buf.seek(0)
    await message.reply_photo(BufferedInputFile(buf.getvalue(), filename="stock_mix.png"), caption=stock_mix.code+' '+stock_mix.name+" created")

@router.message(Command("xqimport"))
async def xqimport(message: Message):
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
    await message.reply_photo(BufferedInputFile(buf.getvalue(), filename="stock_mix.png"), caption=stock_mix.code+' '+stock_mix.name+" created")

@router.message(Command("status"))
async def status(message: Message):
    """
    plot the return rate of given stock mix
    """
    logging.info(f'{message.chat.id}: {message.text}')
    args = argparse_status(message.text)
    if args.help:
        await message.reply(argparse_status.__doc__)
        return 0
    if args.query:
        results = [f for f in os.listdir(data_path) if args.stock_mix_code in f]
        portfolios = []
        for f in results:
            with open(os.path.join(data_path, f), 'rb') as fp:
                portfolios.append(pickle.load(fp))
        await message.reply('\n'.join([portfolio.__repr__() for portfolio in portfolios]))
        return 0
    stock_list = stock_query(keyword=args.stock_mix_code)
    logging.info(f'query result:{stock_list}')
    if len(stock_list) == 1 and type(stock_mix := stock_list[0]) is Stock_mix:
        buf = io.BytesIO()
        if args.ls:
            stock_mix.draw(output=buf)
            buf.seek(0)
            await message.reply_photo(BufferedInputFile(buf.getvalue(), filename="stock_mix.png"), caption=str(stock_mix))
        else:
            # add buffer time for non-trading days
            time_created = stock_mix.create_time.strftime("%Y%m%d")
            time_begin = (stock_mix.create_time - datetime.timedelta(days=9)).strftime("%Y%m%d")
            time_now = datetime.datetime.utcnow().strftime("%Y%m%d %H:%M:%S")
            # stock_data, matrix_close_price = mix_data_collector(stock_mix, time_begin=time_begin)

            try:
                stock_data, matrix_close_price = await mix_data_collector_async(stock_mix, time_begin=time_begin, time_ref='created')
            except QueryError as e:
                await message.reply(f"Failed to collect data: {e}")
                return 1
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
            await message.reply_photo(
                BufferedInputFile(buf.getvalue(), filename="status.png"),
                caption=stock_mix.code+' '+stock_mix.name+\
                                                   "\nCurrent return rate: {:.2%}".format(profit_ratio[-1]))
    else:
        pass

@router.message(Command("now"))
async def now(message: Message, query: CallbackQuery | None = None, command_text: str | None = None):
    text = command_text or message.text
    logging.info(f'{text}')
    try:
        args = argparse_now(text)
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
                time_begin, _ = get_time_range(int(text.split()[2]))
            except IndexError:
                time_begin = stock_mix.create_time.strftime("%Y%m%d")
            datetime_ref = (datetime.datetime.utcnow() - datetime.timedelta(days=30)).strftime("%Y%m%d") # refer to last trading day
            stock_data, matrix_close_price = await mix_data_collector_async(stock_mix, time_begin=datetime_ref)
            time.sleep(random.randint(0,3))
            profit_ratio, stock_profit_ratio = stock_mix.get_profit_ratio(stock_data, matrix_close_price, 
                                                                        date_ref='latest')
            plot_stock_profit(stock_mix, stock_profit_ratio, 
                            title=f'{stock_mix.name} Latest return rate', #TODO add a timestamp
                            output=buf)
            buf.seek(0)
            photo = BufferedInputFile(buf.getvalue(), filename="now.png")
            await message.reply_photo(photo, caption=stock_mix.code+' '+stock_mix.name+\
                                                "\nLatest return rate: {:.2%}".format(profit_ratio[-1]))
        else:
            if stock_info := stock.company_info:
                stock_info = f"[INFO]({stock_info})"
            stock_data, stock_preclose = stock.collect_data_live()
            latest_price = float(list(stock_data['close'])[-1])
            stock_pct = '{:.2%}'.format((latest_price - stock_preclose) / stock_preclose)
            plot_kline(stock_data=stock_data, live=True, preclose=stock_preclose,
                       title=f'Live price of {stock.code} (UTC+8)', plot_type='line', volume=True, macd=False, output=buf)
            buf.seek(0)
            now_caption = (' '.join([stock.code, stock.name, stock_pct, stock_info])).replace('*', r'\*') # A-share sucks!
            photo = BufferedInputFile(buf.getvalue(), filename="now.png")
            if args.md5:
                # Not open to user input, can only result from inline keyboard callback
                await query.message.edit_media(InputMediaPhoto(media=photo, caption=now_caption, parse_mode=ParseMode.MARKDOWN))
            else:
                await message.reply_photo(photo, caption=now_caption, parse_mode=ParseMode.MARKDOWN)
    else:
        # get user's selection from inline keyboard
        keyboard_markup = InlineKeyboardMarkup(inline_keyboard=[])
        for stock in stock_list:
            stock_emoji = _market_emoji[stock.market]
            keyboard_markup.row(InlineKeyboardButton(' '.join([stock_emoji, stock.code, stock.name]), 
                                callback_data=' '.join(filter(None, ['/now', '-e', stock.md5, args.keyword]))))
        # add exit button
        keyboard_markup.row(InlineKeyboardButton('exit', callback_data='exit'))
        await message.reply_photo(_file_id_inline, caption="Find multiple results", reply_markup=keyboard_markup)

@router.callback_query(F.data.startswith('/now'))
async def inline_now_answer_callback_handler(query: CallbackQuery):
    logging.info(f'{query.inline_message_id}: {query.data}')
    await now(query.message, query=query, command_text=query.data)

@router.message(Command("compare"))
async def compare(message: Message):
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
    await message.reply_photo(BufferedInputFile(buf.getvalue(), filename="compare.png"), caption='')
    
# To get photo file_id
# @router.message(F.content_type == types.ContentType.PHOTO)
# async def echo(message):
#     await message.answer(str(message.photo[0].file_id))


async def main():
    try:
        session = AiohttpSession(proxy=config["telegram"]["proxy"])
    except KeyError:
        session = AiohttpSession()
    bot = Bot(token=config["telegram"]["token"], session=session, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
    await dp.start_polling(bot, skip_updates=True)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

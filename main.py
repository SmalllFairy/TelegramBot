# coding: utf-8

# @SbTestChatakaMagicBot

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from TelBot import TelBOT

telbot = TelBOT()

def idle_main(bot, update):
    user_input = update.message.text
    result1, result2 = telbot.handle_input(user_input)
    bot.sendMessage(update.message.chat_id, text="RESULT 1 <question,answer, confidence>" + "\n")
    for text in result1:
        bot.sendMessage(update.message.chat_id, text= text + "\n==========\n")
    bot.sendMessage(update.message.chat_id, text="RESULT 2: <cluster id, cluster name, reference question, "
                                                 "confidence>" + "\n")
    for text in result2:
        bot.sendMessage(update.message.chat_id, text= text + "\n==========\n")

def slash_start(bot, update):
    bot.sendMessage(update.message.chat_id, text="Привет! Введите свой вопрос")

def get_help(bot, update):
    # logger.info('get_help received message: {}'.format(update.message.text))
    help_msg = ('Привет! Я отвечаю на вопросы по банковской тематике.\n'
                'В настоящий момент я поддерживаю следующие команды:\n'
                '/start - начинает наш чат\n'
                '/help - печатает это сообщение').format(
        update.message.from_user.first_name, update.message.from_user.last_name, bot.name)
    bot.sendMessage(update.message.chat_id, text=help_msg)

def main():
    TG_TOKEN = "477886899:AAE2PewVfDwxWYkv0waaESsEXdUt4efucVE"
    updater = Updater(TG_TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", slash_start))
    dp.add_handler(CommandHandler("help", get_help))
    dp.add_handler(MessageHandler(Filters.text, idle_main))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    print("hi!")
    main()


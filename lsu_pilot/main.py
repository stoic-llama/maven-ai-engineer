import os
import logging
from telegram import Update
# from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import numpy as np
from questions import answer_question
import json

from functions import functions, run_function

load_dotenv()  # take environment variables from .env.

tg_bot_token = os.getenv("TG_BOT_TOKEN")
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)


messages = [{
  "role": "system",
  "content": "You are a helpful assistant that answers questions."
}]

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO)

'''
1. UPDATE: object that contains all the information and data 
that are coming from Telegram itself (like the message, 
the user who issued the command, etc) 

2. CONTEXT: object that contains information and data about 
the status of the library itself (like the Bot, the Application, 
the job_queue, etc).

SOURCE: https://github.com/python-telegram-bot/python-telegram-bot/wiki/Extensions---Your-first-Bot
'''
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="I'm a bot, please talk to me!")

# async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
#   messages.append({"role": "user", "content": update.message.text})
#   completion = openai.chat.completions.create(model="gpt-3.5-turbo",
#                                               messages=messages)
#   completion_answer = completion.choices[0].message
#   messages.append(completion_answer)

#   await context.bot.send_message(chat_id=update.effective_chat.id,
#                                  text=completion_answer.content)

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
  messages.append({"role": "user", "content": update.message.text})
  initial_response = openai.chat.completions.create(model="gpt-3.5-turbo",
                                                    messages=messages,
                                                    tools=functions)
  initial_response_message = initial_response.choices[0].message
  messages.append(initial_response_message)
  final_response = None
  tool_calls = initial_response_message.tool_calls
  if tool_calls:
    for tool_call in tool_calls:
      name = tool_call.function.name
      args = json.loads(tool_call.function.arguments)
      response = run_function(name, args)
      print(tool_calls)
      messages.append({
          "tool_call_id": tool_call.id,
          "role": "tool",
          "name": name,
          "content": str(response),
      })
      if name == 'svg_to_png_bytes':
        await context.bot.send_photo(chat_id=update.effective_chat.id,
                                     photo=response)
      # Generate the final response
      final_response = openai.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=messages,
      )
      final_answer = final_response.choices[0].message

      # Send the final response if it exists
      if (final_answer):
        messages.append(final_answer)
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text=final_answer.content)
      else:
        # Send an error message if something went wrong
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text='something wrong happened, please try again')
  #no functions were execute
  else:
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text=initial_response_message.content)   
    

async def mozilla(update: Update, context: ContextTypes.DEFAULT_TYPE):
      answer = answer_question(df, question=update.message.text, debug=True)
      await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


####################

messages = [{
  "role": "system",
  "content": "You are a helpful assistant that answers questions."
}]

CODE_PROMPT = """
Here are two input:output examples for code generation. Please use these and follow the styling for future requests that you think are pertinent to the request.
Make sure All HTML is generated with the JSX flavoring.
// SAMPLE 1
// A Blue Box with 3 yellow cirles inside of it that have a red outline
<div style={{   backgroundColor: 'blue',
  padding: '20px',
  display: 'flex',
  justifyContent: 'space-around',
  alignItems: 'center',
  width: '300px',
  height: '100px', }}>
  <div style={{     backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{     backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{     backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
</div>
"""

messages = [{
  "role": "system",
  "content": "You are a helpful assistant that answers questions."
}, {
  "role": "system",
  "content": CODE_PROMPT
}]


if __name__ == '__main__':
  application = ApplicationBuilder().token(tg_bot_token).build()

  start_handler = CommandHandler('start', start)
  # chat_handler = CommandHandler('chat', chat) #new chat CommandHandler created
  chat_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), chat)
  mozilla_handler = CommandHandler('mozilla', mozilla)
  
  application.add_handler(start_handler)
  application.add_handler(chat_handler) 
  application.add_handler(mozilla_handler)

  application.run_polling()


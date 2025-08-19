from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from transformers import pipeline
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# A dictionary for storing messages passed between ai and human
chats = {}

# Defining the function to return past message if any or create new one
def get_by_session_id(session_id:str):
    if session_id not in chats:
        chats[session_id] = InMemoryChatMessageHistory()
    return chats[session_id]


# A pipeline for the sentiment analysis task
sentiment_pipeline = pipeline(task="sentiment-analysis", model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')

# Defining a function to return the labels for the predicted sentiment
def sentiment_llm(inputs):
    result = sentiment_pipeline(inputs['query'][0])
    return result


# Creating a prompt template to provide instructions, message history and user's query to the model
prompt = ChatPromptTemplate([
    ('system', 'You are a helpful assistant who replies to users based on their sentiments. Do not use any emoji. If sentiment is POSITIVE → be cheerful. If NEGATIVE → be empathetic. If NEUTRAL → be neutral and factual. Sentiment={sentiment}'),
    MessagesPlaceholder(variable_name='conv_history'),
    ('human', '{query}')
])

# Define the model to use as a chat-model
groq_llm = ChatGroq(model='deepseek-r1-distill-llama-70b', reasoning_format='hidden')

# Combining the sentiment-analyser, prompt and the chat-model together
chain = RunnablePassthrough.assign(sentiment=RunnableLambda(sentiment_llm)) | prompt | groq_llm

# Using message history to retain the past conversation knowledge
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key='query',
    history_messages_key='conv_history'
)

# Loop until the user presses quit or exit
while True:
    # Take user's input
    user_input = input('User: ')
    if user_input.lower() in ['quit', 'exit']:
        break
    # Call the model with past conversation_knowledge to get the required response
    response = chain_with_history.invoke({'query': user_input}, {'configurable': {'session_id': 'abc'}})
    print(f"Bot: {response.content}")
import os
from dotenv import load_dotenv
from logger import create_logger
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
load_dotenv()
logger = create_logger(__name__)

def llm_model(llm="groq"):

    if llm == "groq" or llm == "":
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        GROQ_MODEL_NAME=os.getenv("GROQ_MODEL_NAME")
        if not GROQ_MODEL_NAME:
            raise ValueError("GROQ_MODEL_NAME not found in environment variables")
        model = ChatGroq(api_key=GROQ_API_KEY, temperature=0, model_name=GROQ_MODEL_NAME)
        logger.info("Groq model loaded successfully")

    elif llm == "gpt":
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        GPT3_MODEL_NAME=os.getenv("GPT3_MODEL_NAME")
        if not GPT3_MODEL_NAME:
            raise ValueError("GPT3_MODEL_NAME not found in environment variables")
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name=GPT3_MODEL_NAME)
        logger.info("GPT3.5-turbo model loaded successfully")

    elif llm == 'gpt4':
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        GPT4_MODEL_NAME=os.getenv("GPT4_MODEL_NAME")
        if not GPT4_MODEL_NAME:
            raise ValueError("GPT4_MODEL_NAME not found in environment variables")
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name=GPT4_MODEL_NAME)
        logger.info("GPT4o-mini model loaded successfully")
    else:
        logger.error("Invalid LLM model specified")
        raise ValueError("Invalid LLM model specified")

    return model
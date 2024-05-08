"""
This script demonstrates a sophisticated use of the LangChain library to build an agent capable of 
conducting web searches, scraping and summarizing content, and generating Twitter threads based on 
processed information. The agent adheres to a set of research standards, ensuring the information 
provided is factual and well-documented. By exposing the agent as a web service through FastAPI, the 
script makes these capabilities accessible over the web, allowing for a wide range of applications, from automated research to content creation for social media.
"""

import os  # Used for accessing environment variables stored in the system or .env file.
from dotenv import load_dotenv  # Loads environment variables from a .env file into the script, making API keys accessible.

from langchain import PromptTemplate, LLMChain  # LangChain library imports for creating prompt templates and chaining language model operations.
from langchain.agents import initialize_agent, Tool  # For initializing the LangChain agent and defining tools that the agent can use.
from langchain.agents import AgentType  # Enum to specify the type of agent being initialized.
from langchain.chat_models import ChatOpenAI  # For utilizing OpenAI's models within the LangChain framework.
from langchain.prompts import MessagesPlaceholder  # Allows for dynamic insertion of messages into prompts.
from langchain.memory import ConversationSummaryBufferMemory  # Manages memory for the agent, enabling it to remember past interactions.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits large texts into manageable parts for processing.
from langchain.chains.summarize import load_summarize_chain  # Loads a predefined chain for summarizing text.
from langchain.tools import BaseTool  # Base class for creating custom tools within the LangChain framework.
from pydantic import BaseModel, Field  # For creating data models with type validation.
from typing import Type  # Used for type hints, especially when specifying the type of a model or class.
from newspaper3k import Article # For parsing HTML content and extracting information from it.
import requests  # For making HTTP requests to APIs or web pages.
import json  # For encoding and decoding JSON data, especially when interacting with APIs.
from langchain.schema import SystemMessage  # For defining system messages that guide the agent's behavior.
from twit import tweeter  # Custom module assumed to be for interacting with Twitter's API.
from fastapi import FastAPI  # Web framework for creating APIs, making the agent accessible over HTTP.


# Load environment variables for API keys
load_dotenv()
serper_api_key = os.getenv("SERP_API_KEY")

"""
LangChain Agent Initialization and Configuration

The script begins by importing necessary libraries and setting up environment variables for API keys. 
It then defines two primary functionalities as tools: search and scrape_website. These tools are designed 
to perform web searches using the SERP API and scrape content from websites using the Browserless API, 
respectively.

Search Tool: This tool sends a query to the SERP API and returns the search results. It's useful for 
finding relevant web pages or information online based on a user's query.
Scrape Website Tool: This tool takes a URL and an objective as inputs, scrapes the content of the given 
URL, and optionally summarizes the content if it's too lengthy. This is particu
"""

# Define a function to perform a search using the SERP API
# This tool is used for searching the web for information based on a query
def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    return response.text

# Define a function to scrape a website using the Browserless API
# This tool scrapes the content of a website and summarizes it if too large
def scrape_website(objective: str, url: str):
    print("Scraping website using Newspaper3k...")
    # Create an Article object
    article = Article(url)
    
    # Download and parse the article
    article.download()
    article.parse()
    
    text = article.text
    print("CONTENTTTTTT:", text)

    if len(text) > 10000:
        output = summary(objective, text)  # Summarize if text is too long
        return output
    else:
        return text

"""
Summarization Function
A significant part of the script is dedicated to summarizing scraped content. The summary function uses 
the GPT-3.5 model to extract key information from the text, ensuring that only relevant details are 
retained. This function is crucial for processing large amounts of text and presenting it in a 
digestible format.
"""
# Define a function to summarize text using the GPT-3.5 model
# This function extracts key information from the text, ensuring relevance and factual accuracy
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Extract the key information for the following text for {objective}. The text is Scraped data from a website so 
    will have a lot of usless information that doesnt relate to this topic, links, other news stories etc.. 
    Only summarise the relevant Info and try to keep as much factual information Intact
    Do not describe what the webpage is, you are here to get acurate and specific information
    Example of what NOT to do: "Investor's Business Daily: Investor's Business Daily provides news and trends on AI stocks and artificial intelligence. They cover the latest updates on AI stocks and the trends in artificial intelligence. You can stay updated on AI stocks and trends at [AI News: Artificial Intelligence Trends And Top AI Stocks To Watch "
    Here is the text:

    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

"""
LangChain Agent Tools and System Message
The script defines a ScrapeWebsiteTool class, which encapsulates the scraping functionality, making it a 
reusable component within the LangChain framework. The agent is then initialized with the defined tools 
(search and ScrapeWebsiteTool), along with a SystemMessage that outlines the agent's capabilities and 
guidelines for conducting research. This message serves as a prompt to guide the agent's behavior, 
ensuring it adheres to a set of research standards.

Classes Explanation
ScrapeWebsiteInput: This class is a Pydantic model that defines the structure and expected types of 
inputs for the scrape_website function. It ensures that the inputs provided to the function meet the 
expected criteria, enhancing code reliability and reducing runtime errors.

ScrapeWebsiteTool: Inherits from BaseTool, a part of the LangChain library. This class encapsulates the 
functionality for scraping websites as a reusable tool within the LangChain framework. It defines how 
the tool should be executed (_run method) and what inputs it expects (args_schema). This approach 
modularizes the functionality, making it easier to manage and reuse across different parts of the 
application or in future projects.

Query: Another Pydantic model used to define the expected structure of queries received by the FastAPI 
endpoint. It ensures that incoming HTTP requests have the correct format, facilitating error handling
and data validation.
"""

# Pydantic models for input validation
class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")

# Define a tool for scraping websites
class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")

# Initialize the LangChain agent with the defined tools
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

# System message to guide the agent's behavior
system_message = SystemMessage(
content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ Always look at the web first
            7/ Output as much information as possible, make sure your answer is at least 500 WORDS
            8/ Be specific about your reasearch, do not just point to a website and say things can be found here, that what you are for
            

            Example of what NOT to do return these are just a summary of whats on the website an nothing specific, these tell the user nothing!!

            1/WIRED - WIRED provides the latest news, articles, photos, slideshows, and videos related to artificial intelligence. Source: WIRED

            2/Artificial Intelligence News - This website offers the latest AI news and trends, along with industry research and reports on AI technology. Source: Artificial Intelligence News
            """
)

# Additional configurations for the agent
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

# Initialize the LangChain agent with OpenAI's GPT-3.5 model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

# Finalize the agent setup with tools and configurations
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

# Define the FastAPI app for creating an API endpoint
app = FastAPI()

"""
FastAPI Web Service
The script utilizes FastAPI to expose the LangChain agent as a web service. This allows users to send 
queries to the agent via HTTP requests, making the agent's functionalities accessible over the web. 
The endpoint receives a query, processes it through the agent, and returns the generated Twitter 
thread based on the query's content.
"""

# Pydantic model for the API query input
class Query(BaseModel):
    query: str

# API endpoint to process queries through the agent and generate Twitter threads
@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    thread = llm_chain.predict(info = actual_content, topic = query)
    ret = tweetertweet(thread)
    return ret


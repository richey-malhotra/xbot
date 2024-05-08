# This script demonstrates a sophisticated use of the LangChain library to build an agent capable of
# conducting web searches, scraping and summarizing content, and generating Twitter threads based on
# processed information. The agent adheres to a set of research standards, ensuring the information
# provided is factual and well-documented. By exposing the agent as a web service through FastAPI, the
# script makes these capabilities accessible over the web, allowing for a wide range of applications,
# from automated research to content creation for social media.

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
from pydantic import BaseModel, Field, ValidationError  # For creating data models with type validation and handling validation errors.
from typing import Type  # Used for type hints, especially when specifying the type of a model or class.
from newspaper import Article, ArticleException  # For parsing HTML content and extracting information from it, and handling exceptions.
import requests  # For making HTTP requests to APIs or web pages.
from requests.exceptions import RequestException  # For handling exceptions related to HTTP requests.
import json  # For encoding and decoding JSON data, especially when interacting with APIs.
from langchain.schema import SystemMessage  # For defining system messages that guide the agent's behavior.
from twit import tweeter  # Custom module assumed to be for interacting with Twitter's API.
from fastapi import FastAPI, HTTPException  # Web framework for creating APIs, making the agent accessible over HTTP, and handling HTTP exceptions.


# Load environment variables for API keys
load_dotenv()
serper_api_key = os.getenv("SERP_API_KEY")

if not serper_api_key:
    raise ValueError("SERP_API_KEY not found in environment variables.")

# LangChain Agent Initialization and Configuration

# The script begins by importing necessary libraries and setting up environment variables for API keys.
# It then defines two primary functionalities as tools: search and scrape_website. These tools are designed
# to perform web searches using the SERP API and scrape content from websites using the Browserless API,
# respectively.

# Search Tool: This tool sends a query to the SERP API and returns the search results. It's useful for
# finding relevant web pages or information online based on a user's query.
# Scrape Website Tool: This tool takes a URL and an objective as inputs, scrapes the content of the given
# URL, and optionally summarizes the content if it's too lengthy. This is particularly useful for extracting
# relevant information from web pages to support the agent's research capabilities.


# Define a function to perform a search using the SERP API
# This tool is used for searching the web for information based on a query
def search(query: str) -> str:
    """
    Performs a web search using the SERP API based on the provided query.

    Args:
        query (str): The search query.

    Returns:
        str: The search results as a JSON string.

    Raises:
        RequestException: If there's an error while making the HTTP request to the SERP API.
    """
    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        print(response.text)
        return response.text
    except RequestException as e:
        print(f"Error while making the request to SERP API: {e}")
        raise


# Define a function to scrape a website using the Browserless API
# This tool scrapes the content of a website and summarizes it if too large
def scrape_website(objective: str, url: str) -> str:
    """
    Scrapes the content of a website using the Newspaper3k library and summarizes it if the content is too large.

    Args:
        objective (str): The objective or task that users give to the agent.
        url (str): The URL of the website to be scraped.

    Returns:
        str: The scraped content or the summary of the content if it's too large.

    Raises:
        ArticleException: If there's an error while downloading or parsing the article.
    """
    print("Scraping website using Newspaper3k...")
    try:
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
    except ArticleException as e:
        print(f"Error while scraping the website: {e}")
        raise


# Summarization Function
# A significant part of the script is dedicated to summarizing scraped content. The summary function uses
# the GPT-3.5 model to extract key information from the text, ensuring that only relevant details are
# retained. This function is crucial for processing large amounts of text and presenting it in a
# digestible format.


# Define a function to summarize text using the GPT-3.5 model
# This function extracts key information from the text, ensuring relevance and factual accuracy
def summary(objective: str, content: str) -> str:
    """
    Summarizes the provided content using the GPT-3.5 model, extracting key information relevant to the given objective.

    Args:
        objective (str): The objective or task that users give to the agent.
        content (str): The content to be summarized.

    Returns:
        str: The summary of the content, focusing on relevant information.
    """
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Extract the key information for the following text for {objective}. The text is Scraped data from a website so 
    will have a lot of usless information that doesnt relate to this topic, links, other news stories etc.. 
    Only summarise the relevant Info and try to keep as much factual information Intact
    Do not describe what the webpage is, you are here to get acurate and specific information
    Example of what NOT to do: "Investor's Business Daily: Investor's Business Daily provides news and trends on AI 
    stocks and artificial intelligence. They cover the latest updates on AI stocks and the trends in artificial intelligence. 
    You can stay updated on AI stocks and trends at [AI News: Artificial Intelligence Trends And Top AI Stocks To Watch "
    Here is the text:

    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"]
    )

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True,
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


# LangChain Agent Tools and System Message
# The script defines a ScrapeWebsiteTool class, which encapsulates the scraping functionality, making it a
# reusable component within the LangChain framework. The agent is then initialized with the defined tools
# (search and ScrapeWebsiteTool), along with a SystemMessage that outlines the agent's capabilities and
# guidelines for conducting research. This message serves as a prompt to guide the agent's behavior,
# ensuring it adheres to a set of research standards.

# The agent is configured with additional settings, such as extra prompt messages (using MessagesPlaceholder)
# and a ConversationSummaryBufferMemory to enable the agent to remember past interactions. The agent is
# initialized using the OpenAI's GPT-3.5 model, which powers its language understanding and generation capabilities.

# Classes Explanation
# ScrapeWebsiteInput: This class is a Pydantic model that defines the structure and expected types of
# inputs for the scrape_website function. It ensures that the inputs provided to the function meet the
# expected criteria, enhancing code reliability and reducing runtime errors.

# ScrapeWebsiteTool: Inherits from BaseTool, a part of the LangChain library. This class encapsulates the
# functionality for scraping websites as a reusable tool within the LangChain framework. It defines how
# the tool should be executed (_run method) and what inputs it expects (args_schema). This approach
# modularizes the functionality, making it easier to manage and reuse across different parts of the
# application or in future projects.

# Query: Another Pydantic model used to define the expected structure of queries received by the FastAPI
# endpoint. It ensures that incoming HTTP requests have the correct format, facilitating error handling
# and data validation.


# Pydantic models for input validation
class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""

    objective: str = Field(
        description="The objective & task that users give to the agent"
    )
    url: str = Field(description="The url of the website to be scraped")


# Define a tool for scraping websites
class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str) -> str:
        """
        Runs the scrape_website function with the provided objective and URL.

        Args:
            objective (str): The objective or task that users give to the agent.
            url (str): The URL of the website to be scraped.

        Returns:
            str: The scraped content or the summary of the content if it's too large.

        Raises:
            ArticleException: If there's an error while scraping the website.
        """
        try:
            return scrape_website(objective, url)
        except ArticleException as e:
            raise ValueError(f"Error while scraping the website: {e}")

    def _arun(self, url: str):
        raise NotImplementedError(
            "Asynchronous execution is not implemented for this tool."
        )


# Initialize the LangChain agent with the defined tools
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions",
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
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000
)

# Finalize the agent setup with tools and configurations
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

# The MultiQueryRetriever class is designed to handle more complex queries by breaking them down
# into smaller sub-queries. It uses a combination of a query generator and a retriever to
# process the sub-queries and aggregate the results.

# To use the MultiQueryRetriever, you first need to initialize a query generator. The query
# generator is responsible for breaking down the user's query into smaller sub-queries.
# LangChain provides a few pre-built query generators, such as the RefineQueryGenerator and
# the TreeQueryGenerator, but you can also create a custom query generator by inheriting from
# the BaseQueryGenerator class.

# Once you have a query generator, you can initialize the MultiQueryRetriever by passing in
# the query generator and a retriever. The retriever is responsible for processing the sub-queries
# and retrieving relevant documents from your data source. LangChain provides a variety of
# retrievers, such as the ElasticsearchRetriever and the FAISSRetriever, which you can use
# depending on your specific needs.

# After initializing the MultiQueryRetriever, you can use it to process the user's query by
# calling the .get_relevant_documents() method. This method will automatically break down the
# query into sub-queries, process them using the retriever, and aggregate the results.


from langchain.retrievers.multi_query import MultiQueryRetriever

template = """
You are a very experienced ghostwriter who excels at writing Twitter threads.
You will be given a bunch of info below and a topic headline, your job is to use this info and your own knowledge
to write an engaging Twitter thread.
The first tweet in the thread should have a hook and engage with the user to read on.

Here is your style guide for how to write the thread:
1. Voice and Tone:
Informative and Clear: Prioritize clarity and precision in presenting data. 
Phrases like "Research indicates," "Studies have shown," and "Experts suggest" impart a tone of credibility.
Casual and Engaging: Maintain a conversational tone using contractions and approachable language. Pose occasional questions to the reader to ensure engagement.
2. Mood:
Educational: Create an atmosphere where the reader feels they're gaining valuable insights or learning something new.
Inviting: Use language that encourages readers to dive deeper, explore more, or engage in a dialogue.
3. Sentence Structure:
Varied Sentence Lengths: Use a mix of succinct points for emphasis and longer explanatory sentences for detail.
Descriptive Sentences: Instead of directive sentences, use descriptive ones to provide information. E.g., "Choosing a topic can lead to..."
4. Transition Style:
Sequential and Logical: Guide the reader through information or steps in a clear, logical sequence.
Visual Emojis: Emojis can still be used as visual cues
5. Rhythm and Pacing:
Steady Flow: Ensure a smooth flow of information, transitioning seamlessly from one point to the next.
Data and Sources: Introduce occasional statistics, study findings, or expert opinions to bolster claims, and offer links or references for deeper dives.
6. Signature Styles:
Intriguing Introductions: Start tweets or threads with a captivating fact, question, or statement to grab attention.
Question and Clarification Format: Begin with a general question or statement and follow up with clarifying information. 
E.g., "Why is sleep crucial? A study from XYZ University points out..."

Engaging Summaries: Conclude with a concise recap or an invitation for further discussion to keep the conversation going.
Distinctive Indicators for an Informational Twitter Style:

Leading with Facts and Data: Ground the content in researched information, making it credible and valuable.
Engaging Elements: The consistent use of questions and clear, descriptive sentences ensures engagement without leaning heavily on personal anecdotes.
Visual Emojis as Indicators: Emojis are not just for casual conversations; they can be effectively used to mark transitions or emphasize points even in an 
informational context.
Open-ended Conclusions: Ending with questions or prompts for discussion can engage readers and foster a sense of community around the content.

Last instructions:
The twitter thread should be between the length of 3 and 10 tweets 
Each tweet should start with (tweetnumber/total length)
Dont overuse hashtags, only one or two for entire thread.
The first tweet, do not place a number at the start.
When numbering the tweetes Only the tweetnumber out of the total tweets. i.e. (1/9) not (tweet 1/9)
Use links sparingly and only when really needed, but when you do make sure you actually include them AND ONLY PUT THE LINk, dont put brackets around them. 
Only return the thread, no other text, and make each tweet its own paragraph.
Make sure each tweet is lower that 220 chars
Topic Headline:{topic}
Info: {info}
"""

prompt = PromptTemplate(input_variables=["info", "topic"], template=template)

llm = ChatOpenAI(model_name="gpt-4")
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
)

twitapi = tweeter()


def tweetertweet(thread):

    tweets = thread.split("\n\n")

    # check each tweet is under 280 chars
    for i in range(len(tweets)):
        if len(tweets[i]) > 280:
            prompt = f"Shorten this tweet to be under 280 characters: {tweets[i]}"
            tweets[i] = llm.predict(prompt)[:280]
    # give some spacing between sentances
    tweets = [s.replace(". ", ".\n\n") for s in tweets]

    for tweet in tweets:
        tweet = tweet.replace("**", "")

    try:
        response = twitapi.create_tweet(text=tweets[0])
        id = response.data["id"]
        tweets.pop(0)
        for i in tweets:
            print("tweeting: " + i)
            reptweet = twitapi.create_tweet(
                text=i,
                in_reply_to_tweet_id=id,
            )
            id = reptweet.data["id"]
        return "Tweets posted successfully"
    except Exception as e:
        return f"Error posting tweets: {e}"


# Define the FastAPI app for creating an API endpoint
app = FastAPI()


# FastAPI Web Service
# The script utilizes FastAPI to expose the LangChain agent as a web service. This allows users to send 
# queries to the agent via HTTP requests, making the agent's functionalities accessible over the web. 
# The endpoint receives a query, processes it through the agent, and returns the generated Twitter 
# thread based on the query's content.

# Pydantic model for the API query input
class Query(BaseModel):
    query: str


# API endpoint to process queries through the agent and generate Twitter threads
@app.post("/")
def researchAgent(query: Query):
    """
    Processes the provided query through the LangChain agent and generates a Twitter thread based on the query's content.

    Args:
        query (Query): The query object containing the user's input.

    Returns:
        dict: A dictionary containing the generated Twitter thread.

    Raises:
        HTTPException: If there's an error while processing the query or generating the Twitter thread.
    """

    query = query.query
    content = agent({"input": query})
    actual_content = content["output"]
    thread = llm_chain.predict(info=actual_content, topic=query)
    ret = tweetertweet(thread)
    return ret

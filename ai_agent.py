
import os
import time
import json
import uuid
import hashlib
import logging
import datetime
import requests
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Tuple, Union
from bs4 import BeautifulSoup
from functools import lru_cache
from dotenv import load_dotenv
import sqlite3
import redis

# LangChain and LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("research_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Redis for caching (if available)
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", ""),
        decode_responses=True
    )
    redis_available = True
    logger.info("Redis cache initialized successfully")
except Exception as e:
    redis_available = False
    logger.warning(f"Redis cache not available: {e}. Using LRU cache instead.")

# Initialize SQLite for persistent storage
def init_db():
    conn = sqlite3.connect('research_system.db')
    cursor = conn.cursor()

    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS queries (
        id TEXT PRIMARY KEY,
        query TEXT,
        timestamp TEXT,
        model TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS research_results (
        id TEXT PRIMARY KEY,
        query_id TEXT,
        search_results TEXT,
        research_summary TEXT,
        draft_answer TEXT,
        final_response TEXT,
        sources TEXT,
        metrics TEXT,
        FOREIGN KEY (query_id) REFERENCES queries(id)
    )
    ''')

    conn.commit()
    conn.close()
    logger.info("Database initialized")

init_db()

# API Key Management
def get_api_key(key_name, st_input_key=None):
    """Get API key from environment or Streamlit secrets, with UI fallback"""
    key = os.getenv(key_name)

    # If running in Streamlit and key not found in environment
    if not key and st_input_key and 'st' in globals():
        if key_name in st.session_state:
            key = st.session_state[key_name]
        else:
            key = st.text_input(f"Enter your {key_name}:", type="password", key=st_input_key)
            if key:
                st.session_state[key_name] = key

    return key

# Rate Limiter Class
class RateLimiter:
    def __init__(self, calls_per_minute=60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.lock = False

    def wait_if_needed(self):
        """Wait if rate limit is reached"""
        current_time = time.time()
        # Remove calls older than 1 minute
        self.calls = [call for call in self.calls if current_time - call < 60]

        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (current_time - self.calls[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)

        self.calls.append(time.time())

# Cache layer
def get_cache_key(func_name, *args, **kwargs):
    """Generate a cache key from function name and arguments"""
    key_parts = [func_name]
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        elif isinstance(arg, (dict, list, tuple)):
            key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest())

    for k, v in sorted(kwargs.items()):
        key_parts.append(k)
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(str(v))
        elif isinstance(v, (dict, list, tuple)):
            key_parts.append(hashlib.md5(json.dumps(v, sort_keys=True).encode()).hexdigest())

    return hashlib.md5(":".join(key_parts).encode()).hexdigest()

def cache_result(func_name, key, result, ttl=3600):
    """Cache a result with TTL"""
    if redis_available:
        try:
            redis_client.setex(f"research:{key}", ttl, json.dumps(result))
            return True
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            return False
    return False

def get_cached_result(func_name, key):
    """Get a cached result"""
    if redis_available:
        try:
            cached = redis_client.get(f"research:{key}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Redis cache retrieval error: {e}")
    return None

# LRU cache as fallback
@lru_cache(maxsize=100)
def lru_cached_search(query_hash, depth, max_results):
    """LRU cache for search results - just a placeholder for the real implementation"""
    pass  # This function is not directly called, just used to maintain the LRU cache

# Available language models
AVAILABLE_MODELS = {
    "GPT-4o": {"class": ChatOpenAI, "params": {"model": "gpt-4o", "temperature": 0.2}},
    "GPT-3.5-Turbo": {"class": ChatOpenAI, "params": {"model": "gpt-3.5-turbo", "temperature": 0.2}},
    "Claude Opus": {"class": ChatAnthropic, "params": {"model": "claude-3-opus-20240229", "temperature": 0.2}},
    "Claude Sonnet": {"class": ChatAnthropic, "params": {"model": "claude-3-sonnet-20240229", "temperature": 0.2}},
    "Gemini Pro": {"class": ChatGoogleGenerativeAI, "params": {"model": "gemini-pro", "temperature": 0.2}}
}

# State definition
class AgentState(TypedDict):
    query_id: str
    query: str
    model_name: str
    search_results: List[Dict[str, Any]]
    web_content: List[Dict[str, str]]
    research_summary: Optional[str]
    verified_facts: Optional[List[Dict[str, Any]]]
    draft_answer: Optional[str]
    final_response: Optional[str]
    sources: Optional[List[Dict[str, str]]]
    metrics: Optional[Dict[str, Any]]
    errors: List[Dict[str, Any]]

# Tavily Search with rate limiting and caching
tavily_rate_limiter = RateLimiter(calls_per_minute=30)  # Adjust based on your plan

def search_with_tavily(query, search_depth="advanced", max_results=5):
    """Search using Tavily API with caching and rate limiting"""
    cache_key = get_cache_key("tavily_search", query, search_depth, max_results)
    cached_result = get_cached_result("tavily_search", cache_key)

    if cached_result:
        logger.info(f"Cache hit for Tavily search: {query[:30]}...")
        return cached_result

    try:
        tavily_rate_limiter.wait_if_needed()
        tavily_api_key = get_api_key("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("Tavily API key not found")

        tavily_client = TavilyClient(api_key=tavily_api_key)
        results = tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results
        )
        cache_result("tavily_search", cache_key, results, ttl=86400)  # Cache for 24 hours
        return results
    except Exception as e:
        logger.error(f"Tavily search error: {str(e)}")
        # Return empty results or fallback to another search method
        return []

# Web Crawler with error handling and rate limiting
web_rate_limiter = RateLimiter(calls_per_minute=20)  # Be gentle with websites

def crawl_url(url, timeout=10):
    """Crawl a URL with error handling and rate limiting"""
    cache_key = get_cache_key("crawl_url", url)
    cached_result = get_cached_result("crawl_url", cache_key)

    if cached_result:
        logger.info(f"Cache hit for URL: {url[:30]}...")
        return cached_result

    try:
        web_rate_limiter.wait_if_needed()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts, styles, and other unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Extract main content
        main_content = ""

        # Try to find main content area
        main_elem = soup.find('main') or soup.find('article') or soup.find('div', {'id': 'content'})

        if main_elem:
            for paragraph in main_elem.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li']):
                text = paragraph.get_text().strip()
                if text:
                    main_content += text + "\n\n"
        else:
            # Fallback to all paragraphs
            for paragraph in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5']):
                text = paragraph.get_text().strip()
                if text:
                    main_content += text + "\n\n"

        title = soup.title.string if soup.title else url

        result = {
            "url": url,
            "title": title,
            "content": main_content[:10000],  # Limit content size
            "timestamp": datetime.datetime.now().isoformat()
        }

        cache_result("crawl_url", cache_key, result, ttl=604800)  # Cache for 7 days
        return result

    except requests.exceptions.RequestException as e:
        logger.error(f"Error crawling {url}: {str(e)}")
        return {"url": url, "title": url, "content": "", "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error crawling {url}: {str(e)}")
        return {"url": url, "title": url, "content": "", "error": str(e)}

# Initialize language model based on selection
def init_language_model(model_name):
    """Initialize a language model based on name"""
    if model_name not in AVAILABLE_MODELS:
        logger.error(f"Model {model_name} not found, falling back to GPT-3.5-Turbo")
        model_name = "GPT-3.5-Turbo"

    model_info = AVAILABLE_MODELS[model_name]
    model_class = model_info["class"]
    model_params = model_info["params"]

    # Get API keys based on model type
    if model_class == ChatOpenAI:
        api_key = get_api_key("OPENAI_API_KEY")
        if api_key:
            model_params["api_key"] = api_key
    elif model_class == ChatAnthropic:
        api_key = get_api_key("ANTHROPIC_API_KEY")
        if api_key:
            model_params["api_key"] = api_key
    elif model_class == ChatGoogleGenerativeAI:
        api_key = get_api_key("GOOGLE_API_KEY")
        if api_key:
            model_params["api_key"] = api_key

    try:
        return model_class(**model_params)
    except Exception as e:
        logger.error(f"Error initializing model {model_name}: {str(e)}")
        # Fallback to a more reliable model
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# Agent implementations
def research_agent(state: AgentState) -> AgentState:
    """Agent responsible for searching and collecting information"""
    query = state["query"]
    query_id = state["query_id"]
    model_name = state["model_name"]
    errors = state.get("errors", [])

    try:
        # Step 1: Perform initial search with Tavily
        logger.info(f"Research agent starting search for query: {query[:50]}...")
        search_results = search_with_tavily(query)

        if not search_results:
            errors.append({
                "agent": "research",
                "error": "No search results found",
                "timestamp": datetime.datetime.now().isoformat()
            })
            return {**state, "search_results": [], "errors": errors}

        # Step 2: Crawl the top search results to extract detailed content
        web_content = []
        for result in search_results:
            try:
                url = result["url"]
                content_result = crawl_url(url)
                if content_result.get("content"):
                    web_content.append(content_result)
            except Exception as e:
                logger.error(f"Error processing search result: {str(e)}")
                errors.append({
                    "agent": "research",
                    "error": f"Error crawling URL {result.get('url')}: {str(e)}",
                    "timestamp": datetime.datetime.now().isoformat()
                })

        if not web_content:
            errors.append({
                "agent": "research",
                "error": "Failed to extract content from any search results",
                "timestamp": datetime.datetime.now().isoformat()
            })
            return {**state, "search_results": search_results, "web_content": [], "errors": errors}

        # Step 3: Create a research summary
        llm = init_language_model(model_name)

        research_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research agent that synthesizes information from multiple sources.
            Create a comprehensive research summary based on the web content provided.
            Focus on extracting key facts, insights, and diverse perspectives related to the query.
            Include relevant quotes and statistics with proper attribution to sources.
            Structure your summary with clear sections and bullet points for key findings.
            """),
            ("human", "Query: {query}\n\nWeb Content: {web_content_summary}")
        ])

        # Create a summarized version of web content to avoid token limits
        web_content_summary = []
        for content in web_content:
            summary = {
                "url": content["url"],
                "title": content["title"],
                "content": content["content"][:3000]  # Limit content size for each source
            }
            web_content_summary.append(summary)

        research_chain = research_prompt | llm
        research_summary = research_chain.invoke({
            "query": query,
            "web_content_summary": json.dumps(web_content_summary, indent=2)
        }).content

        # Create source list
        sources = []
        for content in web_content:
            sources.append({
                "url": content["url"],
                "title": content["title"]
            })

        # Save intermediate results to database
        conn = sqlite3.connect('research_system.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO research_results (id, query_id, search_results, research_summary, sources) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), query_id, json.dumps(search_results), research_summary, json.dumps(sources))
        )
        conn.commit()
        conn.close()

        # Update state with research results
        return {
            **state,
            "search_results": search_results,
            "web_content": web_content,
            "research_summary": research_summary,
            "sources": sources,
            "errors": errors
        }

    except Exception as e:
        logger.error(f"Research agent error: {str(e)}")
        errors.append({
            "agent": "research",
            "error": f"Research agent failed: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        })
        return {**state, "errors": errors}

def verification_agent(state: AgentState) -> AgentState:
    """Agent responsible for verifying facts and claims"""
    query = state["query"]
    research_summary = state["research_summary"]
    model_name = state["model_name"]
    errors = state.get("errors", [])

    try:
        if not research_summary:
            errors.append({
                "agent": "verification",
                "error": "No research summary to verify",
                "timestamp": datetime.datetime.now().isoformat()
            })
            return {**state, "verified_facts": [], "errors": errors}

        llm = init_language_model(model_name)

        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact verification agent.
            Your job is to identify key claims from the research summary and verify their credibility.
            For each important claim:
            1. Extract the claim
            2. Note the source(s) supporting it
            3. Assess credibility (High, Medium, Low)
            4. Identify if any claims contradict each other from different sources

            Output should be a JSON array of verified facts.
            """),
            ("human", "Query: {query}\n\nResearch Summary: {research_summary}")
        ])

        verification_chain = verification_prompt | llm | JsonOutputParser()

        try:
            verified_facts = verification_chain.invoke({
                "query": query,
                "research_summary": research_summary
            })
        except Exception as e:
            logger.error(f"Error parsing verification output: {str(e)}")
            # Fallback to raw output
            raw_result = verification_prompt | llm
            raw_output = raw_result.invoke({
                "query": query,
                "research_summary": research_summary
            }).content

            # Basic parsing attempt
            try:
                import re
                json_match = re.search(r'\[\s*{.+}\s*\]', raw_output, re.DOTALL)
                if json_match:
                    verified_facts = json.loads(json_match.group(0))
                else:
                    verified_facts = [{"error": "Could not parse verification output", "raw": raw_output}]
            except:
                verified_facts = [{"error": "Could not parse verification output", "raw": raw_output}]

        # Update state with verified facts
        return {
            **state,
            "verified_facts": verified_facts,
            "errors": errors
        }

    except Exception as e:
        logger.error(f"Verification agent error: {str(e)}")
        errors.append({
            "agent": "verification",
            "error": f"Verification agent failed: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        })
        return {**state, "errors": errors}

def draft_agent(state: AgentState) -> AgentState:
    """Agent responsible for drafting answers based on research"""
    query = state["query"]
    query_id = state["query_id"]
    research_summary = state["research_summary"]
    verified_facts = state.get("verified_facts", [])
    model_name = state["model_name"]
    errors = state.get("errors", [])

    try:
        if not research_summary:
            errors.append({
                "agent": "draft",
                "error": "No research summary to draft from",
                "timestamp": datetime.datetime.now().isoformat()
            })
            return {**state, "draft_answer": "", "errors": errors}

        llm = init_language_model(model_name)

        draft_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating clear, comprehensive, and well-structured answers.
            Based on the research summary and verified facts provided, draft a detailed response to the user's query.
            Include key findings, insights, and proper citations to sources where appropriate.
            Structure your answer logically with sections and bullet points when needed.
            Make sure to address different perspectives or contradicting information if present.
            """),
            ("human", """Query: {query}

            Research Summary: {research_summary}

            Verified Facts: {verified_facts}""")
        ])

        draft_chain = draft_prompt | llm
        draft_answer = draft_chain.invoke({
            "query": query,
            "research_summary": research_summary,
            "verified_facts": json.dumps(verified_facts, indent=2) if verified_facts else "No verified facts available."
        }).content

        # Save draft to database
        conn = sqlite3.connect('research_system.db')
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE research_results SET draft_answer = ? WHERE query_id = ?",
            (draft_answer, query_id)
        )
        conn.commit()
        conn.close()

        # Update state with draft answer
        return {
            **state,
            "draft_answer": draft_answer,
            "errors": errors
        }

    except Exception as e:
        logger.error(f"Draft agent error: {str(e)}")
        errors.append({
            "agent": "draft",
            "error": f"Draft agent failed: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        })
        return {**state, "errors": errors}

def controller_agent(state: AgentState) -> Dict[str, Any]:
    """Controls the workflow and determines next steps"""
    query = state["query"]
    model_name = state["model_name"]
    research_summary = state.get("research_summary")
    verified_facts = state.get("verified_facts")
    draft_answer = state.get("draft_answer")
    errors = state.get("errors", [])

    try:
        # If we don't have research yet, go to research agent
        if not research_summary:
            return {"next": "research"}

        # If we have research but no verification, go to verification agent
        elif research_summary and not verified_facts:
            return {"next": "verification"}

        # If we have research and verification but no draft, go to draft agent
        elif research_summary and verified_facts and not draft_answer:
            return {"next": "draft"}

        # If we have all components, generate final response and evaluate
        elif research_summary and draft_answer:
            llm = init_language_model(model_name)

            final_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a final editor responsible for ensuring the highest quality response.
                Review the draft answer, the research summary, and verified facts to ensure accuracy, completeness, and clarity.
                Make any necessary improvements while maintaining the core information and insights.
                Ensure proper citations are included where appropriate.
                Your response should be comprehensive, well-structured, and directly address the user's query.
                """),
                ("human", """Query: {query}

                Draft Answer: {draft_answer}

                Research Summary: {research_summary}

                Verified Facts: {verified_facts}""")
            ])

            final_chain = final_prompt | llm
            final_response = final_chain.invoke({
                "query": query,
                "draft_answer": draft_answer,
                "research_summary": research_summary,
                "verified_facts": json.dumps(verified_facts, indent=2) if verified_facts else "No verified facts available."
            }).content

            # Generate evaluation metrics
            eval_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an evaluation system for research responses.
                Analyze the final response and provide metrics on:
                1. Comprehensiveness (1-10)
                2. Accuracy (1-10)
                3. Source Diversity (1-10)
                4. Clarity and Structure (1-10)
                5. Query Relevance (1-10)

                Output should be a JSON object with these metrics and a brief explanation for each score.
                """),
                ("human", """Query: {query}

                Final Response: {final_response}""")
            ])

            eval_chain = eval_prompt | llm | JsonOutputParser()

            try:
                metrics = eval_chain.invoke({
                    "query": query,
                    "final_response": final_response
                })
            except Exception as e:
                logger.error(f"Error parsing evaluation metrics: {str(e)}")
                metrics = {
                    "error": "Could not generate metrics",
                    "overall_score": 7,  # Default reasonable score
                }

            # Save final results to database
            conn = sqlite3.connect('research_system.db')
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE research_results SET final_response = ?, metrics = ? WHERE query_id = ?",
                (final_response, json.dumps(metrics), state["query_id"])
            )
            conn.commit()
            conn.close()

            # Update state with final response and end the workflow
            return {
                "next": END,
                "final_response": final_response,
                "metrics": metrics,
                "errors": errors
            }

        # Fallback
        errors.append({
            "agent": "controller",
            "error": "Unexpected state in controller",
            "timestamp": datetime.datetime.now().isoformat()
        })
        return {"next": END, "errors": errors}

    except Exception as e:
        logger.error(f"Controller agent error: {str(e)}")
        errors.append({
            "agent": "controller",
            "error": f"Controller agent failed: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        })
        return {"next": END, "errors": errors}

# Create the LangGraph workflow
def create_research_graph():
    # Initialize the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("controller", controller_agent)
    workflow.add_node("research", research_agent)
    workflow.add_node("verification", verification_agent)
    workflow.add_node("draft", draft_agent)

    # Add edges
    workflow.add_edge("controller", "research")
    workflow.add_edge("controller", "verification")
    workflow.add_edge("controller", "draft")
    workflow.add_edge("controller", END)
    workflow.add_edge("research", "controller")
    workflow.add_edge("verification", "controller")
    workflow.add_edge("draft", "controller")

    # Set the entry point
    workflow.set_entry_point("controller")

    return workflow.compile()

# Main research system function
def deep_research_system(query: str, model_name: str = "GPT-3.5-Turbo") -> Dict[str, Any]:
    """Main function to handle user queries and return researched responses"""
    query_id = str(uuid.uuid4())

    # Save query to database
    conn = sqlite3.connect('research_system.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO queries (id, query, timestamp, model) VALUES (?, ?, ?, ?)",
        (query_id, query, datetime.datetime.now().isoformat(), model_name)
    )
    conn.commit()
    conn.close()

    # Initialize the graph
    graph = create_research_graph()

    # Initial state
    initial_state = {
        "query_id": query_id,
        "query": query,
        "model_name": model_name,
        "search_results": [],
        "web_content": [],
        "research_summary": None,
        "verified_facts": None,
        "draft_answer": None,
        "final_response": None,
        "sources": None,
        "metrics": None,
        "errors": []
    }

    # Execute the graph
    try:
        logger.info(f"Starting research for query: {query[:50]}... with model: {model_name}")
        result = graph.invoke(initial_state)
        logger.info(f"Research completed for query: {query[:50]}...")
        return result

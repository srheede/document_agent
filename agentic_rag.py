from __future__ import annotations

import asyncio
import os
import re
import logging
from dataclasses import dataclass
import sys
from typing import Any, List, Dict
from datetime import datetime, timedelta

from dotenv import load_dotenv

from openai import AsyncOpenAI

load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("Error: OPENAI_API_KEY environment variable is not set.")

aclient = AsyncOpenAI(api_key=api_key)  # Used for embeddings
from supabase import create_client, Client

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

# Load environment variables (again, if needed)
load_dotenv()

# Set log level from env variable; default to ERROR.
log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.ERROR))

# Set model (default to gpt-4o-mini)
llm = os.getenv("LLM_MODEL", "gpt-4o-mini")
model = (
    OpenAIModel(
        llm,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
    if os.getenv("OPEN_ROUTER_API_KEY", None)
    else OpenAIModel(llm)
)

# Create Supabase client
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# Define a dependency dataclass for our agent
@dataclass
class DocumentationDeps:
    supabase: Client

# Helper: Get embedding for a text string using OpenAI async API.
async def get_embedding(text: str) -> List[float]:
    logging.debug(f"Getting embedding for text (first 30 chars): {text[:30]}...")
    try:
        response = await aclient.embeddings.create(model="text-embedding-3-small", input=text)
        embedding = response.data[0].embedding
        logging.debug("Successfully retrieved embedding.")
        return embedding
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return [0] * 1536

# Updated system prompt with detailed instructions for tool flow.
system_prompt = (
    "You are Arag, a documentation assistant with access to a knowledgebase of documentation chunks. "
    "Your name is Arag and the user’s name is Stef. You have the following tools available:\n\n"
    "1. determine_query_type(query: str) -> str\n"
    "   - Determines if a query is 'general' or 'documentation'.\n\n"
    "2. get_documentation_titles() -> List[Dict[str,Any]]\n"
    "   - Returns a list of documentation entries (each with an id and title) representing available document chunks.\n\n"
    "3. get_documentation_summary(doc_id: int) -> str\n"
    "   - Provides additional context for a documentation entry identified by its id. Use this if the title alone is not clear.\n\n"
    "4. retrieve_relevant_documentation(query: str, match_count: int) -> List[Dict[str,Any]]\n"
    "   - Retrieves the full content of all the documentation chunks that have been identified as relevant.\n\n"
    "5. resolve_time_period(query: str) -> str\n"
    "   - If the query mentions a time period (e.g., 'last week'), returns the corresponding date range (e.g., '10 March - 16 March 2025').\n\n"
    "When answering a documentation-specific query, first use get_documentation_titles to list all available titles. "
    "Then, if it is unclear whether a chunk is relevant based on its title alone, use get_documentation_summary for further context. "
    "If the query mentions a time period, use resolve_time_period to compute the corresponding dates. "
    "After determining which chunks are relevant, use retrieve_relevant_documentation to fetch their full content. "
    "Finally, formulate your answer solely based on the retrieved documentation chunks. "
    "If the query is general, answer it directly using general world knowledge."
    "IMPORTANT: Provide only the final answer without including any tool call metadata or IDs."
)

# Create the main agent.
doc_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=DocumentationDeps,
    retries=2
)

# Tool: determine_query_type – classify the query using a temporary agent.
@doc_agent.tool
async def determine_query_type(ctx: RunContext[DocumentationDeps], query: str) -> str:
    logging.debug(f"Determining query type for: {query}")
    classification_prompt = (
        "You are a query classification agent. Your task is to decide whether a user's question should be "
        "answered using general world knowledge or by retrieving specific documented personal information. "
        "If the query refers to particular dates, events, or personal experiences that are likely recorded in a "
        "knowledgebase, classify it as 'documentation'. Otherwise, classify it as 'general'. "
        "Respond with exactly one word: 'general' or 'documentation'.\n\n"
        f"Query: {query}"
    )
    classification_agent = Agent(
        model,
        system_prompt=classification_prompt,
        retries=2
    )
    result = await classification_agent.run("")
    classification = result.data.strip().lower()
    logging.debug(f"Query type determined: {classification}")
    return classification if classification in ["general", "documentation"] else "general"

# Tool: retrieve_relevant_documentation – get matching documentation chunks.
@doc_agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[DocumentationDeps], query: str, match_count: int = 10) -> List[Dict[str, Any]]:
    logging.debug(f"Retrieving relevant documentation for query: {query}")
    embedding = await get_embedding(query)
    try:
        response = ctx.deps.supabase.rpc("match_site_pages", {
            "query_embedding": embedding,
            "match_count": match_count,
            "filter": {}
        }).execute()
        if hasattr(response, "error") and response.error is not None:
            logging.error(f"Supabase RPC error: {response.error}")
            return []
        logging.debug(f"Retrieved {len(response.data)} documentation chunks.")
        return response.data
    except Exception as e:
        logging.error(f"Error in retrieve_relevant_documentation: {e}")
        return []

# Tool: get_documentation_titles – list distinct documentation entries.
@doc_agent.tool
async def get_documentation_titles(ctx: RunContext[DocumentationDeps]) -> List[Dict[str, Any]]:
    logging.debug("Fetching documentation titles.")
    try:
        response = ctx.deps.supabase.table("site_pages").select("id, title").execute()
        if hasattr(response, "data"):
            data = response.data
        elif isinstance(response, dict):
            data = response.get("data", [])
        else:
            data = []
        seen = set()
        unique_titles = []
        for row in data:
            title = row.get("title")
            if title and title not in seen:
                seen.add(title)
                unique_titles.append(row)
        logging.debug(f"Fetched {len(unique_titles)} unique titles.")
        return unique_titles
    except Exception as e:
        logging.error(f"Error in get_documentation_titles: {e}")
        return []

# Tool: get_documentation_summary – retrieve a documentation summary by its id.
@doc_agent.tool
async def get_documentation_summary(ctx: RunContext[DocumentationDeps], doc_id: int) -> str:
    logging.debug(f"Fetching documentation summary for id: {doc_id}")
    try:
        response = ctx.deps.supabase.table("site_pages").select("summary").eq("id", doc_id).limit(1).execute()
        if hasattr(response, "data"):
            data = response.data
        elif isinstance(response, dict):
            data = response.get("data", [])
        else:
            data = []
        if data:
            summary = data[0].get("summary", "")
            logging.debug("Successfully retrieved summary.")
            return summary
        logging.debug("No summary found.")
        return ""
    except Exception as e:
        logging.error(f"Error in get_documentation_summary: {e}")
        return ""

# New Tool: resolve_time_period – use OpenAI to determine the date range for a time period in the query.
@doc_agent.tool
async def resolve_time_period(ctx: RunContext[DocumentationDeps], query: str) -> str:
    logging.debug(f"Resolving time period for query using OpenAI: {query}")
    now = datetime.now()
    current_date = now.strftime("%d %B %Y %H:%M:%S")
    prompt = (
        f"You are a helpful assistant specialized in understanding time periods. "
        f"Given the current date and time, and a query mentioning a time period, determine the date range that the query refers to. "
        f"Current date and time: {current_date}\n\n"
        f"Query: {query}\n\n"
        "Output only the date range in the format 'StartDate - EndDate' (for example, '10 March 2025 - 16 March 2025'). "
        "If the query does not clearly refer to a time period, output 'No specific time period identified.'"
    )
    # Create a temporary agent using the prompt
    time_agent = Agent(
        model,
        system_prompt=prompt,
        retries=2
    )
    try:
        result = await time_agent.run("")
        resolved = result.data.strip()
        logging.debug(f"Resolved time period: {resolved}")
        return resolved
    except Exception as e:
        logging.error(f"Error resolving time period with OpenAI: {e}")
        return "No specific time period identified."

# Interactive chat loop using the agent.
async def interactive_chat():
    print("Hello! I am your Agentic RAG. How can I help you today?")
    while True:
        try:
            query = input("\nYou: ").strip()
            if any(word in query.lower() for word in ["bye", "exit", "quit"]):
                print("Thanks for using Agentic RAG! Goodbye!")
                break

            logging.debug(f"Received query: {query}")
            run_result = await doc_agent.run(query, deps=DocumentationDeps(supabase=supabase))
            answer = run_result.data if hasattr(run_result, "data") else run_result
            logging.debug(f"Agent raw output: {run_result}")
            print("\nAgent:", answer)
        except KeyboardInterrupt:
            print("\nThanks for using Agentic RAG! Goodbye!")
            break
        except Exception as e:
            logging.error(f"Error in chat loop: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(interactive_chat())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.debug("Program ended")

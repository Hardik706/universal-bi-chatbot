from pydantic import BaseModel, Field
from typing import Literal
import logging
import sys
import os
# Add parent directory to sys.path if not present to enable clean imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import get_llm

logger = logging.getLogger(__name__)

class RouteIntent(BaseModel):
    """
    Structured classification result from the intent router.
    """
    route: Literal["sql", "rag", "hybrid"] = Field(
        description="The target route. Choose 'sql' for structured database data, 'rag' for unstructured text/document searches, or 'hybrid' for combined structured and unstructured questions."
    )
    reasoning: str = Field(
        description="A brief explanation for why this route was selected."
    )

def route_query(query: str) -> RouteIntent:
    """
    Determines the intent of the user query: 'sql', 'rag', or 'hybrid'.
    Uses structured LLM outputs to guarantee schema alignment.
    """
    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(RouteIntent)
        
        system_prompt = (
            "You are an expert query router for an Enterprise Business Intelligence platform.\n"
            "Your task is to analyze the user's query and classify it into one of three destinations:\n\n"
            "1. 'sql': The question is about structured, numerical, transactional, or schema-based data "
            "stored in a relational database (e.g. sales numbers, client counts, list of products, aggregations, trend lines).\n"
            "2. 'rag': The question is about unstructured texts, user manuals, policy guidelines, operational PDFs, "
            "or documentation features (e.g. how a product works, corporate guidelines, installation steps).\n"
            "3. 'hybrid': The question requires both structured data and unstructured text. "
            "For example: 'Get the total sales of product X and summarize its user manual setup instructions.'\n\n"
            "Provide the output following the requested schema precisely."
        )
        
        messages = [
            ("system", system_prompt),
            ("user", query)
        ]
        
        response = structured_llm.invoke(messages)
        logger.info(f"Routed query: '{query}' -> {response.route} (Reason: {response.reasoning})")
        return response
        
    except Exception as e:
        logger.exception("Failed to route query using LLM. Falling back to SQL route.")
        # Safe fallback default
        return RouteIntent(
            route="sql",
            reasoning=f"Fallback triggered due to error: {str(e)}"
        )

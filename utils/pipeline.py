# pipeline.py

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jVector
from langchain.prompts import PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

# Import your feature extraction function
from utils.txt_feature_extractor import extract_features_from_sample_battery_from_text

# --- Load environment variables ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing!")
if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
    raise ValueError("Neo4j credentials are missing!")

# --- Initialize Neo4j Connection ---
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# --- Initialize LLM and Embeddings ---
llm = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.0
)
embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# --- Initialize Neo4j Vector Store for Semantic Similarity ---
vectorstore = Neo4jVector(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    embedding=embedder
)

# --- (Optional) Example Prompts for Battery Data ---
examples = [
    {
        "question": "Which battery has the highest total cycles?",
        "query": "MATCH (b:Battery) RETURN b.battery_id, b.total_cycles ORDER BY b.total_cycles DESC LIMIT 1"
    },
    # ... add additional examples as needed ...
]
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embedder,
    vectorstore,
    k=5,
    input_keys=["question"]
)

# --- Define the Prompt Template ---
CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher query to extract battery-related data from a Neo4j database.
Schema:
{schema}

Instructions:
- The schema is provided as key-value pairs.
- Identify the battery property mentioned in the question.
- Use the provided numeric values appropriately.
- Your response must be a comma-separated list of battery IDs only (no additional text).

Examples:
# Which battery has the highest total cycles?
MATCH (b:Battery) RETURN b.battery_id, b.total_cycles ORDER BY b.total_cycles DESC LIMIT 1

# Find batteries similar to one with slope_last_500_cycles = -0.000385
MATCH (b:Battery) WHERE abs(b.slope_last_500_cycles - (-0.000385)) < 0.0001 RETURN b.battery_id, b.slope_last_500_cycles

The query is:
{query}
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "query"],
    template=CYPHER_GENERATION_TEMPLATE
)

# --- Create the GraphCypherQAChain instance ---
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True,
)

def run_pipeline(user_query, uploaded_file):
    """
    Run the KG RAG pipeline given a user query and an uploaded battery file.
    
    :param user_query: The user query string.
    :param uploaded_file: A file-like object or a string containing battery data.
    :return: The raw LLM response.
    """
    # Read file content (if it's a file-like object)
    file_content = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")
    
    try:
        features = extract_features_from_sample_battery_from_text(file_content)
        schema_to_use = "\n".join(f"{key}: {value}" for key, value in features.items())
    except Exception as e:
        print(f"Error extracting features from file: {e}. Using fallback schema.")
        try:
            schema_to_use = str(graph.schema)
        except Exception:
            schema_to_use = "Battery nodes with properties like battery_id, total_cycles, slopes, etc."
    
    # Invoke the chain with the extracted schema and the user query
    try:
        response = chain.invoke({"schema": schema_to_use, "query": user_query})
    except Exception as e:
        print(f"Error invoking chain: {e}. Returning empty response for scoring.")
        response = {"result": ""}
    
    return response


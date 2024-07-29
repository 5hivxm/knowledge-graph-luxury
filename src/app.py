import streamlit as st
from langchain_community.chat_models import ChatOpenAI
import os
from langchain.prompts.prompt import PromptTemplate
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from neo4j import GraphDatabase
from py2neo import Graph, Node, Relationship
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components

def process_query(fewshot_cypher_chain, question):
    result = fewshot_cypher_chain(question)
    response_structured = result['result']
    generated_code = result['intermediate_steps'][1]['context']
    return response_structured, generated_code


st.title("Luxury Dashboard")
st.write("Ask a question about the dataset below! To use this app, you need to provide an OpenAI API key.")

openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key

if not openai_api_key:
    st.error("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    question = st.selectbox("Select a Question", 
                            ["Which brand offers the highest-priced product in the dataset?",
                            "What is the average price for Gucci products compared to Balenciaga products?",
                            "Which product category has the highest demand overall?",
                            "How does the price of Gucci Men's Shoes compare to the price of Balenciaga Men's Shoes?",
                            "What is the price difference between the most expensive and the least expensive product in the dataset?",
                            "How does the demand for products correlate with their prices across different brands and product categories?",
                            "Using a knowledge graph, identify the relationship between product cost, competitor price, and demand. How do these factors influence each other?",
                            "Analyze the pricing strategy of Gucci compared to its competitors. What insights can be drawn about their market positioning and competitive advantage?",
                            "Using a knowledge graph, map out the relationships between brands, products, costs, prices, and demand. How can this information be used to optimize pricing and marketing strategies for luxury brands?"],
                            placeholder="Question?", disabled=not openai_api_key)

llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    temperature=0,
    model="gpt-3.5-turbo"
)

url = st.secrets.neo4j.NEO4J_URI
username = st.secrets.neo4j.NEO4J_USERNAME
password = st.secrets.neo4j.NEO4J_PASSWORD
graph = Neo4jGraph(url, username, password)
FEWSHOT_CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about luxury brands and their products.
Convert the user's question based on the schema.
For questions that contain 's, convert to \'s
For questions that contain knowledge graph, return schema to plot it

If no context is returned, do not attempt to answer the question.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Examples:

What is the average price for Gucci products compared to Burberry products?:
MATCH (b:BRAND)-[s:SELLS]->(p:PRODUCT)
WHERE b.name = 'Gucci' OR b.name = 'Burberry'
RETURN b.name, AVG(s.price) as average_price

Analyze the pricing strategy of Gucci compared to its competitors. What insights can be drawn about their market positioning and competitive advantage?:
MATCH (b:BRAND)-[s:SELLS]->(p:PRODUCT)<-[sa:SELLS]-(c:COMPETITOR)-[:COMPETES_WITH]->(b)
WHERE b.name='Gucci'
WITH b, c, AVG(s.price) as avg_price, AVG(sa.price) as avg_competitor_price
RETURN b.name, c.name, avg_price, avg_competitor_price

Using a knowledge graph, map out the relationships between brands, products, costs, prices, and demand. How can this information be used to optimize pricing and marketing strategies for luxury brands?:
MATCH (b:BRAND)-[s:SELLS]->(p:PRODUCT)
RETURN b, s, s.cost, s.price, s.demand, p


Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Question: {question}
"""

FEWSHOT_CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question", "schema"],
    validate_template=True,
    template=FEWSHOT_CYPHER_GENERATION_TEMPLATE
)

fewshot_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=llm,
    qa_llm=ChatOpenAI(temperature=0, api_key=openai_api_key),
    graph=graph, verbose=True,
    cypher_prompt = FEWSHOT_CYPHER_GENERATION_PROMPT, return_intermediate_steps=True
)

response_structured, generated_code = process_query(fewshot_cypher_chain, question)
#st.write(response_structured)


G = nx.MultiDiGraph()

# Add nodes and edges to the graph
for index in generated_code:
    brand = None
    product = None
    rel = None
    if index.get('b'):
        brand = index.get('b')['name']
        G.add_node(brand)
    elif index.get('b.name'):
        brand = index.get('b.name')
        G.add_node(brand)

    if index.get('c'):
        comp = index.get('c')['name']
        G.add_node(comp)
        if brand is not None:
            rel = 'COMPETES_WITH'
            G.add_edge(brand, comp, label=rel)
    elif index.get('c.name'):
        comp = index.get('c.name')
        G.add_node(comp)
        if brand is not None:
            rel = 'COMPETES_WITH'
            G.add_edge(brand, comp, label=rel)

    if index.get('s'):
        rel = index.get('s')[1]
        
    if index.get('p'):
        product = index.get('p')['name']
        G.add_node(product)
        if brand is not None:
            G.add_edge(brand, product, label=rel)
    elif index.get('p.name'):
        product = index.get('p.name')
        G.add_node(product)
        if brand is not None:
            G.add_edge(brand, product, label=rel)


edge_labels = {}
for u, v, data in G.edges(data=True):
    label = data['label']
    if (u, v) not in edge_labels:
        edge_labels[(u, v)] = label
    else:
        edge_labels[(u, v)] += f", {label}"

fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Display the plot in Streamlit
st.title("Graph Visualization")
st.pyplot(fig)

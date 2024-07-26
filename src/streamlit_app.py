import streamlit as st
import os
from openai import OpenAI
from neo4j import GraphDatabase
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from streamlit_agraph import Config
import networkx as nx
from matplotlib import pyplot as plt

def process_query(query):
    result = cypher_chain(query)
    intermediate_steps = result['intermediate_steps']
    final_answer = result['result']
    generated_cypher = intermediate_steps[0]['query']
    response_structured = final_answer    
    return response_structured, generated_cypher

def display_graph(query):
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    with driver.session() as session:
        result = session.run(query)
        G = nx.Graph()
        for record in result:
            brand = record.get('b.name')  # Use .get() to handle missing keys
            product = record.get('p.name')
            competitor = record.get('c.name')
            if brand and product:  # Only add nodes/edges if both Brand and Product exist
                G.add_node(brand)
                G.add_node(product)
                G.add_edge(brand, product, label='SELLS')
                if competitor:  # Add competitor node and edge only if present
                    G.add_node(competitor)
                    G.add_edge(product, competitor, label='SELLS')

        # Visualize the graph
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'))
        plt.show()
        st.pyplot(plt)

st.title("Luxury Dashboard")
st.write("Ask a question about the dataset below! To use this app, you need to provide an OpenAI API key.")

os.environ["NEO4J_URI"] = "neo4j+s://09516404.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "xTyNa-R6nR9NHjpxYGdLWOUFW3jHwzHUHkrU9yk7b2E"

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"])

node_types = ['BRAND', 'PRODUCT', 'COMPETITOR']
relationship_types = ['SELLS', 'COMPETES_WITH']

openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")


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

    client = OpenAI(api_key=openai_api_key)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    cypher_chain = GraphCypherQAChain.from_llm(
        cypher_llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', api_key=openai_api_key),
        qa_llm=ChatOpenAI(temperature=0, api_key=openai_api_key),
        graph=graph,
        verbose=True,
        return_intermediate_steps=True)

    response_structured, generated_cypher = process_query(question)
    config = Config(height=600, width=800, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6",
        node={'color': 'blue'}, link={'color': 'grey'})
    st.chat_message("assistant").write(response_structured)
    display_graph(generated_cypher)

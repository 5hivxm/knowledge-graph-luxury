from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
from openai import OpenAI
import streamlit as st
import getpass
import os
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
import graphviz as graphviz
import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph
from py2neo import Graph
import scipy

st.title("Meetup Dashboard")
st.write(
    "Ask a question about the dataset below! "
    "To use this app, you need to provide an OpenAI API key. "
)

driver = GraphDatabase.driver(st.secrets.neo4j.uri, auth=(st.secrets.neo4j.user, st.secrets.neo4j.password))

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["NEO4J_URI"]=st.secrets.neo4j.uri
os.environ["NEO4J_USERNAME"]=st.secrets.neo4j.user
os.environ["NEO4J_PASSWORD"]=st.secrets.neo4j.password

graph = Neo4jGraph()
uploaded_file = graph
with open('src/file.cypher', 'r') as file:
    luxury_schema = file.read()

graph.query(luxury_schema)
graph.refresh_schema()


question = st.selectbox("Select a Question", 
                          ["Which brand offers the highest-priced product in the dataset?",
                           "What is the average demand for Gucci products compared to Balenciaga products?",
                           "Which product category has the highest demand overall?",
                           "How does the price of Gucci Men's Shoes compare to the price of Balenciaga Men's Shoes?",
                           "What is the price difference between the most expensive and the least expensive product in the dataset?",
                           "How does the demand for products correlate with their prices across different brands and product categories?",
                           "Using a knowledge graph, identify the relationship between product cost, competitor price, and demand. How do these factors influence each other?",
                           "Analyze the pricing strategy of Gucci compared to its competitors. What insights can be drawn about their market positioning and competitive advantage?",
                           "Using a knowledge graph, map out the relationships between brands, products, costs, prices, and demand. How can this information be used to optimize pricing and marketing strategies for luxury brands?"],
                           placeholder="Question?", disabled=not openai_api_key)

if question:
    chain = GraphCypherQAChain.from_llm(
        ChatOpenAI(temperature=0), graph=graph, verbose=True, 
        return_intermediate_steps=True, validate_cypher=True
    )

    result = chain.invoke({"query": question})
    st.write(f"Final answer: {result['result']}")
    query = result['intermediate_steps'][0]['query']
    query = """MATCH (n) RETURN n"""
    # Use neo4j driver to execute the query

    with driver.session() as session:
        result = session.run(query)
        
        # Create a NetworkX graph
        G = nx.Graph()
        for record in result:
            node1 = record['n']
            G.add_node(node1.element_id, label=node1.labels)

        # Draw the graph
        fig, ax = plt.subplots()
        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
        st.pyplot(fig)
        #nx.draw(G, pos, with_labels=True, labels=labels, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')

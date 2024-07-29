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

# Returns the answer and generated Cypher query
def process_query(cypher_chain, query):
    result = cypher_chain(query)
    st.write(result)
    intermediate_steps = result['intermediate_steps']
    final_answer = result['result']
    generated_cypher = intermediate_steps[0]['query']
    response_structured = final_answer    
    return response_structured, generated_cypher

# Displays the graph
def display_graph(query):
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    with driver.session() as session:
        result = session.run(query)
        G = nx.Graph()
        for record in result:
            brand = record.get('b.name') if record.get('b.name') else record.get('Brand')
            product = record.get('p.name') if record.get('p.name') else record.get('Product')
            competitor = record.get('c.name') if record.get('c.name') else record.get('Competitor')
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


    cypher_chain = GraphCypherQAChain.from_llm(
        cypher_llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', api_key=openai_api_key),
        qa_llm=ChatOpenAI(temperature=0, api_key=openai_api_key),
        graph=graph, verbose=True, return_intermediate_steps=True)

    response_structured, generated_cypher = process_query(cypher_chain, question)

    config = Config(height=600, width=800, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6",
        node={'color': 'blue'}, link={'color': 'grey'})
    # Display answer and graph
    st.chat_message("assistant").write(response_structured)
    display_graph(generated_cypher)

import streamlit as st
import os
from openai import OpenAI
from neo4j import GraphDatabase
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from streamlit_agraph import agraph, Node, Edge, Config

def process_query(query):
    result = cypher_chain(query)
    intermediate_steps = result['intermediate_steps']
    final_answer = result['result']
    generated_cypher = intermediate_steps[0]['query']
    response_structured = final_answer
    
    nodes, edges = fetch_graph_data(direct_cypher_query=generated_cypher, intermediate_steps=intermediate_steps)
    
    return response_structured, nodes, edges

def fetch_graph_data(direct_cypher_query=None, intermediate_steps=None):
    if direct_cypher_query:
        context = intermediate_steps[1]['context']
        nodes, edges = process_graph_result(context)
    else:
        cypher_query = construct_cypher_query(node_types, relationship_types)
        with GraphDatabase.driver(os.environ["NEO4J_URI"], 
                                  auth=(os.environ["NEO4J_USERNAME"], 
                                        os.environ["NEO4J_PASSWORD"])).session() as session:
            result = session.run(cypher_query)
            nodes, edges = process_graph_result_select(result)
    
    return nodes, edges

def construct_cypher_query(node_types, rel_types):
    node_clauses = []
    for node_type in node_types:
        node_clauses.append(f"(p:{node_type})-[r]->(n) ")

    rel_clauses = []
    for rel_type in rel_types:
        rel_clauses.append(f"type(r)='{rel_type}' ")

    if rel_clauses:
        rel_match = " OR ".join(rel_clauses)
        query = f"MATCH {' OR '.join(node_clauses)} WHERE {rel_match} RETURN p, r, n"
    else:
        query = f"MATCH {' OR '.join(node_clauses)} RETURN p, r, n"
    
    return query

def process_graph_result(result):
    nodes = []
    edges = []
    node_names = set()

    for record in result: 
        p_name = record.get('p.name')
        b_name = record.get('b.name')

        if p_name and p_name not in node_names:
            nodes.append(Node(id=p_name, label=p_name, size=5, shape="circle"))
            node_names.add(p_name)
        if b_name and b_name not in node_names:
            nodes.append(Node(id=b_name, label=b_name, size=5, shape="circle"))
            node_names.add(b_name)

        relationship_label = record.get('type(r)')
        if p_name and b_name and relationship_label:
            edges.append(Edge(source=p_name, target=b_name, label=relationship_label))
 
    return nodes, edges

def process_graph_result_select(result):
    nodes = []
    edges = []
    node_names = set()

    for record in result:
        p = record.get('p')
        n = record.get('n')
        p_name = p.get('name') if p else None
        n_name = n.get('name') if n else None

        if p_name and p_name not in node_names:
            nodes.append(Node(id=p_name, label=p_name, size=5, shape="circle"))
            node_names.add(p_name)
        if n_name and n_name not in node_names:
            nodes.append(Node(id=n_name, label=n_name, size=5, shape="circle"))
            node_names.add(n_name)

        r = record.get('r')
        if p_name and n_name and r:
            relationship_label = r.get('type')
            if 'date' in r:
                relationship_label = f"{r.get('type')} ({r.get('date')})"
            edges.append(Edge(source=p_name, target=n_name, label=relationship_label))
    
    return nodes, edges

st.title("Meetup Dashboard")
st.write("Ask a question about the dataset below! To use this app, you need to provide an OpenAI API key.")

os.environ["NEO4J_URI"] = st.secrets.neo4j.uri
os.environ["NEO4J_USERNAME"] = st.secrets.neo4j.user
os.environ["NEO4J_PASSWORD"] = st.secrets.neo4j.password

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"])

node_types = ['BRAND', 'PRODUCT', 'COMPETITOR']
relationship_types = ['SELLS', 'COMPETES_WITH']

openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")

question = st.selectbox("Select a Question", 
                        ["Which brand offers the highest-priced product in the dataset?",
                         "What is the average demand for Gucci products compared to Burberry products?",
                         "Which product category has the highest demand overall?",
                         "How does the price of Gucci Men's Shoes compare to the price of Balenciaga Men's Shoes?",
                         "What is the price difference between the most expensive and the least expensive product in the dataset?",
                         "How does the demand for products correlate with their prices across different brands and product categories?",
                         "Using a knowledge graph, identify the relationship between product cost, competitor price, and demand. How do these factors influence each other?",
                         "Analyze the pricing strategy of Gucci compared to its competitors. What insights can be drawn about their market positioning and competitive advantage?",
                         "Using a knowledge graph, map out the relationships between brands, products, costs, prices, and demand. How can this information be used to optimize pricing and marketing strategies for luxury brands?"],
                        placeholder="Question?", disabled=not openai_api_key)

if not openai_api_key:
    st.error("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    client = OpenAI(api_key=openai_api_key)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    cypher_chain = GraphCypherQAChain.from_llm(
        cypher_llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', api_key=openai_api_key),
        qa_llm=ChatOpenAI(temperature=0, api_key=openai_api_key),
        graph=graph,
        verbose=True,
        return_intermediate_steps=True)

    response_structured, nodes, edges = process_query(question)
    config = Config(
        height=600,
        width=800,
        directed=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        node={'color': 'blue'},
        link={'color': 'grey'}
    )
    st.chat_message("assistant").write(response_structured)
    agraph(nodes=nodes, edges=edges, config=config)

#DEBUG EDGES

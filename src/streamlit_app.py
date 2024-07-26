import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
import getpass
import os
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# Show title and description.

st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
)





# Initialize Neo4jGraph
write_query()
graph = Neo4jGraph()
with open('file.cypher', 'r') as file:
    luxury_query = file.read()

graph.query(luxury_query)
graph.refresh_schema()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
uploaded_file = graph

# Ask the user for a question via `st.text_area`.
question = st.text_area(
    "Now ask a question?",
    placeholder="Question?",
    disabled=not uploaded_file,
)

if question:
    # Process the uploaded file and question.
    chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=True)
    response = chain.invoke({"query": question})
    st.write_stream(response)

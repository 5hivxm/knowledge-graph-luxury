import streamlit as st
import graphviz as graphviz
from neo4j_controller import Neo4jController
st.title("Meetup Dashboard")

n4j = Neo4jController(st.streamlit.secrets.neo4j.uri,
                      st.streamlit.secrets.neo4j.user,
                      st.streamlit.secrets.neo4j.password)

brands = n4j.get_brands()
brand_name = st.selectbox("Select a brand", brands)

if brand_name is not None:
    # Number of products
    num_products = n4j.num_products(brand_name)

    # Display metrics
    st.metric("Number of Products", num_products)

    brands = n4j.get_brands()
    products = n4j.get_products(brand_name)

    # Sample of graph_data = [{'attendee': 'Jason K', 'skill': 'Swift (Apple programming language)'}, {'attendee': 'Jason K', 'skill': 'Kotlin'}, {'attendee': 'Jason K', 'skill': 'Dart'}]

    graph = graphviz.Digraph()
    for product in products:
        graph.edge(brand_name, product['product'])

    st.graphviz_chart(graph)
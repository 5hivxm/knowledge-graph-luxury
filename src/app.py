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

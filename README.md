
1. Install the requirements
   ```
   $ pip install -r requirements.txt
   ```

2. Download Neo4j Desktop
- Create LocalDBMS, copy credentials (uri, username, password) into .streamlit/secrets.toml file

3. Open Neo4jBrowser
- Copy and Paste src/file.cypher contents into Query to create Graph

4. Enter OpenAI key into .streamlit/secrets.toml file

5. Run the app
   ```
   $ streamlit run src/streamlit_app.py
   ```

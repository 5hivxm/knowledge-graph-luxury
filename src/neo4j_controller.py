from neo4j import GraphDatabase, unit_of_work

@unit_of_work(timeout=5)
def _get_brands(tx):
    result = tx.run("""
        MATCH (b:BRAND)
        RETURN b.name as name
    """)
    return [record["name"] for record in result.data()]

@unit_of_work(timeout=5)
def _num_products(tx, brand_name):
    result = tx.run("""
        MATCH (b:BRAND {name: $brand_name})-[:HAS_PRODUCT]->(p:PRODUCT)
        RETURN count(distinct p) as num_products
    """, brand_name=brand_name)
    return result.data()[0]["num_products"]

@unit_of_work(timeout=5)
def _get_products(tx, brand_name):
    result = tx.run("""
        MATCH (b:BRAND {name: $brand_name})-[:HAS_PRODUCT]->(p:PRODUCT)
        RETURN p.name as product
    """, brand_name=brand_name)
    return result.data()

class Neo4jController:
    
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(
                self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def get_brands(self):
        assert self.__driver is not None, "Driver not initialized!"
        try:
            with self.__driver.session() as session:
                return session.execute_read(_get_brands)
        except Exception as e:
            print("get_brands failed:", e)

    def num_products(self, brand_name):
        assert self.__driver is not None, "Driver not initialized!"
        try:
            with self.__driver.session() as session:
                return session.execute_read(_num_products, brand_name)
        except Exception as e:
            print("num_products failed:", e)

    def get_products(self, brand_name):
        assert self.__driver is not None, "Driver not initialized!"
        try:
            with self.__driver.session() as session:
                return session.execute_read(_get_products, brand_name)
        except Exception as e:
            print("get_products failed:", e)
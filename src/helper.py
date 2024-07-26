"""
CREATE (:Brand {name: 'Gucci'})
CREATE (:Brand {name: 'Burberry'})
CREATE (:Brand {name: 'Prada'})
CREATE (:Brand {name: 'Versace'})
CREATE (:Competitor {name: 'Balenciaga'})
CREATE (:Competitor {name: 'Hermes'})
CREATE (:Competitor {name: 'Chanel'})
CREATE (:Competitor {name: 'Dior'})

LOAD CSV WITH HEADERS FROM 'file:///src/luxury_real_data.csv' AS row
MERGE (brand:Brand {name: row.Brand})
MERGE (competitor:Competitor {name: row.Competitor})
MERGE (product:Product {name: row.Product, cost: toFloat(row.Cost), price: toFloat(row.Price), demand: toInteger(row.Demand), competitorPrice: toFloat(row.CompetitorPrice)})
MERGE (brand)-[:OFFERS]->(product)
MERGE (product)-[:COMPETES_WITH]->(competitor)
"""
# VIVA PREPARATION - Flight Price Tracker System

**Student Name:** Abdullah Asif  
**Course:** Advanced Database Systems  
**Project:** Flight Price Tracker with Hybrid Search

---

## TABLE OF CONTENTS

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Database Design](#database-design)
4. [Code Explanation - Line by Line](#code-explanation)
5. [Search Implementation](#search-implementation)
6. [API Design](#api-design)
7. [Common Viva Questions & Answers](#common-viva-questions)

---

## PROJECT OVERVIEW

### What does this project do?

This project has two main parts:

**Part 1:** A system that tracks flight ticket prices over time
- We store prices of flights at different time intervals (every 15 days)
- For example: If a flight is on Jan 1, 2026, we track its price starting 6 months before
- So we have price on July 1, July 15, July 30, etc.

**Part 2:** APIs to search and retrieve this data
- We built REST APIs using FastAPI
- We can search flights using different methods (keyword, semantic, hybrid)
- We can get price history for any flight

### Why MongoDB?

MongoDB is a NoSQL database that stores data in JSON-like documents. We chose it because:

1. **Flexible Schema:** Each flight can have different number of price points
2. **Array Storage:** Price history is an array - MongoDB handles this well
3. **Good for Time-Series:** Price history is time-series data
4. **Easy to Work With:** Python's pymongo library makes it simple

### Why FastAPI?

FastAPI is a modern Python web framework. We chose it over Flask because:

1. **Automatic Documentation:** Creates interactive API docs at /docs
2. **Type Validation:** Automatically validates request parameters
3. **Fast:** As the name suggests, it's faster than Flask
4. **Modern:** Uses Python type hints
5. **Your sir taught it:** Expected in the project

---

## SYSTEM ARCHITECTURE

```
User Request
    ↓
FastAPI Application (main.py)
    ↓
Search Utils (search_utils.py) ← Uses sentence-transformers for embeddings
    ↓
MongoDB Database (flight_tracker_db)
    ↓
Response to User
```

### Components:

1. **generate_data.py:** Creates fake but realistic flight price data
2. **seed_database.py:** Puts data into MongoDB and creates embeddings
3. **main.py:** FastAPI application with all API endpoints
4. **search_utils.py:** Functions for keyword, semantic, and hybrid search

---

## DATABASE DESIGN

### Database Name: `flight_tracker_db`
### Collection Name: `flights`

### Document Schema:

```json
{
  "_id": ObjectId("..."),              // MongoDB's unique ID
  "flight_id": "FL1000",               // Our custom ID
  "route": {
    "origin": "LHE",                   // Lahore airport code
    "destination": "JED"               // Jeddah airport code
  },
  "airline": "PIA",                    // Airline name
  "departure_date": "2026-04-15",      // When flight departs
  "base_price": 650,                   // Starting base price
  "price_history": [                   // Array of price records
    {
      "date": "2025-10-15",
      "price": 520,
      "currency": "USD"
    },
    {
      "date": "2025-10-30",
      "price": 545,
      "currency": "USD"
    }
  ],
  "tracking_interval_days": 15,        // How often we track (15 days)
  "embedding": [0.123, -0.456, ...],   // Vector for semantic search
  "created_at": "2025-10-22 14:30:00"  // When record was created
}
```

### Why This Schema?

1. **route as nested object:** Keeps origin and destination together
2. **price_history as array:** Natural way to store time-series data
3. **embedding field:** Needed for semantic search (384-dimensional vector)
4. **flight_id:** Easy to reference (instead of using MongoDB's _id)

### Indexes Created:

```python
collection.create_index([("route.origin", ASCENDING)])
collection.create_index([("route.destination", ASCENDING)])
collection.create_index([("airline", ASCENDING)])
collection.create_index([("departure_date", ASCENDING)])
collection.create_index([("route.origin", TEXT), ("route.destination", TEXT), ("airline", TEXT)])
```

**Why indexes?**
- Make searches faster (like index in a book)
- Without index: MongoDB scans all documents (slow)
- With index: MongoDB directly finds relevant documents (fast)

---

## CODE EXPLANATION - LINE BY LINE

### File 1: generate_data.py

This file creates fake flight data that looks realistic.

```python
import random
from datetime import datetime, timedelta
import json
```

**Line by line:**
- `import random`: To generate random numbers (for prices)
- `from datetime import datetime, timedelta`: To work with dates
- `import json`: To save data in JSON format

```python
def generate_flight_prices():
    routes = [
        {"from": "LHE", "to": "JED", "airline": "PIA"},
        {"from": "LHE", "to": "DXB", "airline": "Emirates"},
        # ... more routes
    ]
```

**What this does:**
- Defines a function called `generate_flight_prices`
- Creates a list of 10 different flight routes
- Each route has origin, destination, and airline

```python
    all_flights_data = []
    
    for idx, route in enumerate(routes):
        days_ahead = random.randint(90, 270)
        flight_date = datetime.now() + timedelta(days=days_ahead)
```

**Line by line:**
- `all_flights_data = []`: Empty list to store all flight data
- `for idx, route in enumerate(routes)`: Loop through each route
  - `idx` is the index (0, 1, 2...)
  - `route` is the current route dictionary
- `days_ahead = random.randint(90, 270)`: Random number between 90-270
  - This means flight is 3 to 9 months in the future
- `flight_date = datetime.now() + timedelta(days=days_ahead)`: Calculate flight date

```python
        base_price = random.randint(400, 1500)
```

**What this does:**
- Sets a base price between $400 to $1500
- Different routes have different base prices (distance effect)

```python
        tracking_start_date = flight_date - timedelta(days=180)
```

**What this does:**
- Start tracking 180 days (6 months) before flight
- If flight is on June 1, start tracking from December 1

```python
        price_history = []
        current_date = tracking_start_date
        
        while current_date <= datetime.now():
```

**What this does:**
- Create empty list for price history
- Start from tracking_start_date
- Loop until we reach today's date
- This generates all historical price points

```python
            days_until_flight = (flight_date - current_date).days
            
            time_factor = 1 + (180 - days_until_flight) / 500
            
            random_factor = random.uniform(0.85, 1.15)
            
            price = int(base_price * time_factor * random_factor)
```

**Line by line explanation:**
- `days_until_flight`: How many days left until flight
- `time_factor`: Makes price increase as flight date approaches
  - When far away: time_factor is close to 1
  - When near: time_factor is higher (price increases)
- `random_factor`: Random fluctuation between 0.85 and 1.15 (±15%)
- `price`: Final price = base_price × time_factor × random_factor

**Example calculation:**
- base_price = 500
- 120 days before flight: time_factor = 1 + (180-120)/500 = 1.12
- random_factor = 1.05
- price = 500 × 1.12 × 1.05 = $588

```python
            price_history.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "price": price,
                "currency": "USD"
            })
            
            current_date += timedelta(days=15)
```

**What this does:**
- Add this price point to price_history array
- `strftime("%Y-%m-%d")`: Format date as "2025-10-15"
- Move to next tracking date (15 days later)

```python
        flight_doc = {
            "flight_id": f"FL{1000 + idx}",
            "route": {
                "origin": route["from"],
                "destination": route["to"]
            },
            "airline": route["airline"],
            "departure_date": flight_date.strftime("%Y-%m-%d"),
            "base_price": base_price,
            "price_history": price_history,
            "tracking_interval_days": 15,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        all_flights_data.append(flight_doc)
```

**What this does:**
- Create a complete flight document
- `f"FL{1000 + idx}"`: Creates flight IDs like FL1000, FL1001, FL1002...
- Organize all data into a dictionary
- Add this flight to all_flights_data list

```python
    return all_flights_data
```

**What this does:**
- Return the list of all 10 flight documents

```python
if __name__ == "__main__":
    print("Generating flight price data...")
    
    flights = generate_flight_prices()
    
    with open("flights_data.json", "w") as f:
        json.dump(flights, f, indent=2)
    
    print(f"Generated {len(flights)} flights data")
    print("Data saved to flights_data.json")
```

**What this does:**
- `if __name__ == "__main__"`: Only runs when we execute this file directly
- Call the generate_flight_prices function
- `with open(...)`: Open a file for writing
- `json.dump(...)`: Save the data as JSON with nice formatting (indent=2)
- Print success messages

---

### File 2: seed_database.py

This file puts the generated data into MongoDB.

```python
from pymongo import MongoClient, ASCENDING, TEXT
from sentence_transformers import SentenceTransformer
import json
```

**Imports:**
- `pymongo`: Library to work with MongoDB from Python
- `MongoClient`: Class to connect to MongoDB
- `ASCENDING, TEXT`: Constants for creating indexes
- `SentenceTransformer`: Model to create embeddings (vectors)
- `json`: To read the JSON file we created

```python
def get_database():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["flight_tracker_db"]
    return db
```

**Line by line:**
- `MongoClient("mongodb://localhost:27017/")`: Connect to MongoDB
  - `localhost`: MongoDB is on same computer
  - `27017`: Default MongoDB port
- `client["flight_tracker_db"]`: Get/create database named "flight_tracker_db"
- Return the database object

```python
def create_embeddings_for_flights(flights):
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
```

**What this does:**
- Load the sentence-transformers model
- `all-MiniLM-L6-v2`: A small, fast model (only 80MB)
- This model converts text to vectors (384 numbers)

**Why embeddings?**
- For semantic search, we need to convert text to numbers
- Similar meanings = similar vectors
- Example: "Lahore to Dubai" and "LHE to DXB" have similar embeddings

```python
    for flight in flights:
        text = f"{flight['route']['origin']} to {flight['route']['destination']} {flight['airline']}"
        
        embedding = model.encode(text).tolist()
        
        flight['embedding'] = embedding
    
    return flights
```

**Line by line:**
- Loop through each flight
- `text = f"..."`: Create description like "LHE to JED PIA"
- `model.encode(text)`: Convert text to vector (returns numpy array)
- `.tolist()`: Convert numpy array to Python list (for JSON)
- Add embedding to the flight dictionary

**Example:**
- Input text: "LHE to JED PIA"
- Output: [0.123, -0.456, 0.789, ..., 0.234] (384 numbers)

```python
def seed_database():
    print("Connecting to MongoDB...")
    db = get_database()
    
    collection = db["flights"]
    
    collection.delete_many({})
    print("Cleared existing data")
```

**What this does:**
- Connect to MongoDB database
- Get the "flights" collection (like a table)
- `delete_many({})`: Delete all existing documents
  - `{}` means "match everything"
  - This clears old data before inserting new

```python
    print("Loading flight data from JSON...")
    with open("flights_data.json", "r") as f:
        flights = json.load(f)
```

**What this does:**
- Open the JSON file we created earlier
- `json.load(f)`: Read JSON and convert to Python list

```python
    flights = create_embeddings_for_flights(flights)
```

**What this does:**
- Add embeddings to each flight document
- This takes a few seconds because the model needs to process text

```python
    print("Inserting data into MongoDB...")
    result = collection.insert_many(flights)
    print(f"Inserted {len(result.inserted_ids)} flight records")
```

**What this does:**
- `insert_many(flights)`: Insert all flight documents at once
- Returns result object with inserted IDs
- Print how many documents were inserted

```python
    print("Creating indexes...")
    
    collection.create_index([("route.origin", ASCENDING)])
    collection.create_index([("route.destination", ASCENDING)])
    collection.create_index([("airline", ASCENDING)])
    collection.create_index([("departure_date", ASCENDING)])
    
    collection.create_index([
        ("route.origin", TEXT),
        ("route.destination", TEXT),
        ("airline", TEXT)
    ])
```

**What this does:**
- Create indexes to make searches faster

**Types of indexes:**
1. **ASCENDING indexes:** For exact matching and sorting
   - When we search "route.origin = LHE", MongoDB uses this index
2. **TEXT index:** For keyword search
   - When we search "Emirates Dubai", MongoDB uses this index

**Why multiple indexes?**
- Different types of queries need different indexes
- Having proper indexes makes queries 100x faster

```python
    print("Database seeding completed successfully!")
```

**What this does:**
- Print success message

---

### File 3: search_utils.py

This file contains all search functions.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
```

**What this does:**
- Import required libraries
- Load the embedding model once globally
- This saves time (don't reload model for each search)

#### Function 1: keyword_search

```python
def keyword_search(collection, query, limit=10):
    results = collection.find(
        {"$text": {"$search": query}},
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(limit)
```

**Line by line:**
- `collection.find(...)`: Search in MongoDB collection
- `{"$text": {"$search": query}}`: Use text search on the query
  - This searches in origin, destination, and airline (our TEXT index)
- `{"score": {"$meta": "textScore"}}`: Get relevance score
  - MongoDB calculates how well each document matches
- `.sort([("score", ...)])`: Sort by relevance (best matches first)
- `.limit(limit)`: Return only top results

**Example:**
- Query: "Emirates Dubai"
- MongoDB finds all documents containing "Emirates" or "Dubai"
- Ranks them by relevance
- Returns top 10

```python
    results_list = list(results)
    
    if results_list:
        max_score = max([r.get('score', 0) for r in results_list])
        if max_score > 0:
            for r in results_list:
                r['keyword_score'] = r.get('score', 0) / max_score
    
    return results_list
```

**What this does:**
- Convert MongoDB cursor to list
- Normalize scores to 0-1 range
  - Find the maximum score
  - Divide each score by max_score
  - Now all scores are between 0 and 1
- Add normalized score as 'keyword_score'

**Why normalize?**
- Makes it easier to combine with semantic scores
- Both will be in same range (0-1)

#### Function 2: semantic_search

```python
def semantic_search(collection, query, limit=10):
    query_embedding = model.encode(query).tolist()
```

**What this does:**
- Convert the search query to a vector
- Example: "cheap flights to Bangkok" → [0.234, -0.567, ...]

```python
    all_docs = list(collection.find({}))
```

**What this does:**
- Get all documents from collection
- `{}` means no filter (get everything)

**Note:** In a real large-scale system, we would use vector databases like Pinecone or Weaviate for faster vector search. But for 10 documents, this is fine.

```python
    for doc in all_docs:
        if 'embedding' in doc:
            doc_embedding = doc['embedding']
            similarity = cosine_similarity(query_embedding, doc_embedding)
            doc['semantic_score'] = similarity
        else:
            doc['semantic_score'] = 0
```

**Line by line:**
- Loop through each document
- Check if document has embedding
- Calculate similarity between query and document embeddings
- Store similarity as 'semantic_score'

```python
    all_docs.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
    
    return all_docs[:limit]
```

**What this does:**
- Sort documents by semantic_score (highest first)
  - `lambda x: x.get('semantic_score', 0)`: Get semantic_score from each doc
  - `reverse=True`: Descending order (highest first)
- Return top 'limit' documents

#### Function 3: cosine_similarity

```python
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return float(dot_product / (norm1 * norm2))
```

**What is cosine similarity?**
- Measures how similar two vectors are
- Returns value between 0 and 1
  - 1 = identical
  - 0 = completely different

**Mathematical formula:**
```
similarity = (A · B) / (||A|| × ||B||)
```

Where:
- A · B = dot product (multiply corresponding elements and sum)
- ||A|| = length/magnitude of vector A
- ||B|| = length/magnitude of vector B

**Example:**
- vec1 = [1, 2, 3]
- vec2 = [2, 4, 6]
- dot_product = (1×2) + (2×4) + (3×6) = 2 + 8 + 18 = 28
- norm1 = √(1² + 2² + 3²) = √14 ≈ 3.74
- norm2 = √(2² + 4² + 6²) = √56 ≈ 7.48
- similarity = 28 / (3.74 × 7.48) ≈ 1.0

**Line by line:**
- `np.array(vec1)`: Convert Python list to numpy array
- `np.dot(vec1, vec2)`: Calculate dot product
- `np.linalg.norm(vec1)`: Calculate vector length
- Check if either norm is 0 (avoid division by zero)
- Return the cosine similarity

#### Function 4: hybrid_search

```python
def hybrid_search(collection, query, alpha=0.5, limit=10):
    keyword_results = keyword_search(collection, query, limit=limit*2)
    semantic_results = semantic_search(collection, query, limit=limit*2)
```

**What this does:**
- Run both keyword and semantic search
- Get 2× the limit (to have more candidates)
- We'll combine and re-rank them

**Why 2× limit?**
- Some documents might appear in only one search
- We want enough candidates to choose from

```python
    combined_scores = {}
    
    for doc in keyword_results:
        doc_id = str(doc['_id'])
        combined_scores[doc_id] = {
            'doc': doc,
            'keyword_score': doc.get('keyword_score', 0),
            'semantic_score': 0
        }
```

**What this does:**
- Create dictionary to store combined scores
- Key = document ID
- Value = dictionary with document and scores
- Initialize with keyword scores, semantic score = 0

```python
    for doc in semantic_results:
        doc_id = str(doc['_id'])
        if doc_id in combined_scores:
            combined_scores[doc_id]['semantic_score'] = doc.get('semantic_score', 0)
        else:
            combined_scores[doc_id] = {
                'doc': doc,
                'keyword_score': 0,
                'semantic_score': doc.get('semantic_score', 0)
            }
```

**What this does:**
- Add semantic scores
- If document already exists (was in keyword results), just update semantic_score
- If document is new (only in semantic results), create new entry

**Result:**
- Now every document has both keyword_score and semantic_score
- Some might be 0 (if appeared in only one search)

```python
    for doc_id in combined_scores:
        k_score = combined_scores[doc_id]['keyword_score']
        s_score = combined_scores[doc_id]['semantic_score']
        
        hybrid_score = alpha * k_score + (1 - alpha) * s_score
        combined_scores[doc_id]['hybrid_score'] = hybrid_score
```

**This is the CORE formula:**

```
hybrid_score = α × keyword_score + (1-α) × semantic_score
```

Where α (alpha) is between 0 and 1:
- α = 0: Pure semantic (hybrid_score = semantic_score)
- α = 0.5: Balanced (hybrid_score = 0.5 × keyword + 0.5 × semantic)
- α = 1: Pure keyword (hybrid_score = keyword_score)

**Example calculation:**
- keyword_score = 0.8
- semantic_score = 0.6
- alpha = 0.5
- hybrid_score = 0.5 × 0.8 + 0.5 × 0.6 = 0.4 + 0.3 = 0.7

**Another example:**
- keyword_score = 0.8
- semantic_score = 0.6
- alpha = 0.7 (more weight to keyword)
- hybrid_score = 0.7 × 0.8 + 0.3 × 0.6 = 0.56 + 0.18 = 0.74

```python
    sorted_results = sorted(
        combined_scores.values(),
        key=lambda x: x['hybrid_score'],
        reverse=True
    )
```

**What this does:**
- Sort all documents by hybrid_score
- Highest hybrid_score first

```python
    final_results = []
    for item in sorted_results[:limit]:
        doc = item['doc']
        doc['keyword_score'] = round(item['keyword_score'], 3)
        doc['semantic_score'] = round(item['semantic_score'], 3)
        doc['hybrid_score'] = round(item['hybrid_score'], 3)
        final_results.append(doc)
    
    return final_results
```

**What this does:**
- Take top 'limit' documents
- Add all three scores to each document
- Round scores to 3 decimal places (cleaner output)
- Return final results

#### Function 5: optimize_alpha (BONUS)

```python
def optimize_alpha(collection, query, relevance_feedback):
    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_alpha = 0.5
    best_score = 0
```

**What this does:**
- Try 11 different alpha values (0.0 to 1.0)
- Initialize best_alpha as 0.5 (balanced)
- Initialize best_score as 0

**Purpose:** Find which alpha value gives best results for this specific query

```python
    for alpha in alpha_values:
        results = hybrid_search(collection, query, alpha=alpha, limit=10)
        
        relevant_count = 0
        for i, doc in enumerate(results):
            doc_id = str(doc['_id'])
            if doc_id in relevance_feedback:
                position_weight = 1 / (i + 1)
                relevant_count += position_weight
```

**Line by line:**
- Try each alpha value
- Do hybrid search with this alpha
- Count how many relevant documents appear in results
- `enumerate(results)`: Get both index (i) and document
- Check if this document was marked relevant by user
- `position_weight = 1 / (i + 1)`: Earlier positions get more weight
  - Position 0: weight = 1/1 = 1.0
  - Position 1: weight = 1/2 = 0.5
  - Position 2: weight = 1/3 = 0.33
  - Position 3: weight = 1/4 = 0.25

**Why position weight?**
- Users care more about top results
- A relevant document at position 0 is better than at position 9

**Example:**
- User marked FL1000 and FL1005 as relevant
- With alpha=0.3, results are: [FL1005, FL1002, FL1000, ...]
- relevant_count = 1.0 (FL1005 at position 0) + 0.33 (FL1000 at position 2) = 1.33

```python
        if relevant_count > best_score:
            best_score = relevant_count
            best_alpha = alpha
    
    return best_alpha
```

**What this does:**
- If this alpha gives better results than previous best
- Update best_score and best_alpha
- After trying all alphas, return the best one

**Example output:**
- Alpha 0.0: score = 0.83
- Alpha 0.3: score = 1.33 ← Best!
- Alpha 0.5: score = 1.0
- Alpha 1.0: score = 0.5
- Returns: 0.3

---

### File 4: main.py (FastAPI Application)

This is the main file with all APIs.

```python
from fastapi import FastAPI, HTTPException, Query
from pymongo import MongoClient
from typing import Optional, List
from datetime import datetime
import json
from search_utils import keyword_search, semantic_search, hybrid_search, optimize_alpha
```

**Imports:**
- `FastAPI`: Main class for creating API
- `HTTPException`: To raise errors (like 404 Not Found)
- `Query`: To define query parameters with validation
- `Optional, List`: Type hints from typing module
- `datetime`: For working with dates
- Import our search functions

```python
app = FastAPI(
    title="Flight Price Tracker API",
    description="API to track and search flight ticket prices over time",
    version="1.0.0"
)
```

**What this does:**
- Create FastAPI application instance
- Set title, description, version
- This info appears in the automatic documentation at /docs

```python
client = MongoClient("mongodb://localhost:27017/")
db = client["flight_tracker_db"]
collection = db["flights"]
```

**What this does:**
- Connect to MongoDB (once when app starts)
- Get database and collection
- These are global variables (accessible in all functions)

```python
def clean_doc(doc):
    if doc:
        doc['_id'] = str(doc['_id'])
        if 'embedding' in doc:
            del doc['embedding']
    return doc
```

**What this does:**
- MongoDB's _id is ObjectId type (can't convert to JSON)
- Convert _id to string
- Remove embedding (it's very long and not needed in response)

**Why remove embedding?**
- Embedding has 384 numbers
- Makes response very large
- User doesn't need to see it

#### API Endpoint 1: Root

```python
@app.get("/")
def root():
    return {
        "message": "Flight Price Tracker API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "get_all_flights": "/flights",
            ...
        }
    }
```

**What this does:**
- `@app.get("/")`: Decorator that makes this function an API endpoint
- When user visits `http://localhost:8000/`
- This function runs and returns a dictionary
- FastAPI automatically converts dictionary to JSON

**What is a decorator?**
- `@app.get("/")` is a decorator
- It "decorates" (wraps) the function below it
- Tells FastAPI: "When someone makes GET request to '/', call this function"

#### API Endpoint 2: Get All Flights

```python
@app.get("/flights")
def get_all_flights(limit: int = Query(10, description="Number of flights to return")):
    flights = list(collection.find({}).limit(limit))
    flights = [clean_doc(f) for f in flights]
    
    return {
        "count": len(flights),
        "flights": flights
    }
```

**Line by line:**
- `@app.get("/flights")`: This function handles GET requests to /flights
- `limit: int = Query(10, ...)`: Query parameter with default value 10
  - User can call: `/flights` (uses default 10)
  - Or: `/flights?limit=20` (uses 20)
- `collection.find({})`: Find all documents (no filter)
- `.limit(limit)`: Return only 'limit' documents
- Clean each document
- Return count and list of flights

**What is Query?**
- It's from FastAPI
- Defines a query parameter (part of URL after ?)
- Provides validation and documentation

#### API Endpoint 3: Get Flight by ID

```python
@app.get("/flights/{flight_id}")
def get_flight_by_id(flight_id: str):
    flight = collection.find_one({"flight_id": flight_id})
    
    if not flight:
        raise HTTPException(status_code=404, detail=f"Flight {flight_id} not found")
    
    return clean_doc(flight)
```

**What this does:**
- `{flight_id}`: Path parameter (part of URL path)
  - Example: `/flights/FL1000` → flight_id = "FL1000"
- `find_one(...)`: Find one document matching the filter
- If not found, raise 404 error
- Otherwise, return the flight

**What is HTTPException?**
- Raises an HTTP error
- `status_code=404`: Not Found error
- FastAPI automatically converts this to proper error response

#### API Endpoint 4: Get Price History (MAIN ENDPOINT FOR PART 1)

```python
@app.get("/flights/{flight_id}/prices")
def get_price_history(flight_id: str):
    flight = collection.find_one({"flight_id": flight_id})
    
    if not flight:
        raise HTTPException(status_code=404, detail=f"Flight {flight_id} not found")
    
    return {
        "flight_id": flight['flight_id'],
        "route": f"{flight['route']['origin']} → {flight['route']['destination']}",
        "airline": flight['airline'],
        "departure_date": flight['departure_date'],
        "tracking_interval_days": flight['tracking_interval_days'],
        "price_history": flight['price_history'],
        "current_price": flight['price_history'][-1] if flight['price_history'] else None,
        "lowest_price": min(flight['price_history'], key=lambda x: x['price']) if flight['price_history'] else None,
        "highest_price": max(flight['price_history'], key=lambda x: x['price']) if flight['price_history'] else None
    }
```

**This is the main API for Part 1 of your project!**

**What it returns:**
1. Basic flight info (id, route, airline, date)
2. Full price history (list of all price points)
3. Current price (last price in history)
4. Lowest price (minimum price)
5. Highest price (maximum price)

**Line by line:**
- `flight['price_history'][-1]`: Last element in price_history array
- `min(flight['price_history'], key=lambda x: x['price'])`: Find item with minimum price
  - `key=lambda x: x['price']`: Compare by price field
- `max(...)`: Similar to min, finds maximum price

**Example response:**
```json
{
  "flight_id": "FL1000",
  "route": "LHE → JED",
  "airline": "PIA",
  "departure_date": "2026-04-15",
  "tracking_interval_days": 15,
  "price_history": [
    {"date": "2025-10-15", "price": 520, "currency": "USD"},
    {"date": "2025-10-30", "price": 545, "currency": "USD"}
  ],
  "current_price": {"date": "2025-10-30", "price": 545, "currency": "USD"},
  "lowest_price": {"date": "2025-10-15", "price": 520, "currency": "USD"},
  "highest_price": {"date": "2025-10-30", "price": 545, "currency": "USD"}
}
```

#### API Endpoint 5: Search by Route

```python
@app.get("/search/route")
def search_by_route(
    origin: str = Query(..., description="Origin airport code (e.g., LHE)"),
    destination: str = Query(..., description="Destination airport code (e.g., JED)"),
    airline: Optional[str] = Query(None, description="Airline name (optional)"),
    date: Optional[str] = Query(None, description="Departure date YYYY-MM-DD (optional)")
):
    query = {
        "route.origin": origin.upper(),
        "route.destination": destination.upper()
    }
    
    if airline:
        query["airline"] = {"$regex": airline, "$options": "i"}
    
    if date:
        query["departure_date"] = date
    
    flights = list(collection.find(query))
    
    if not flights:
        raise HTTPException(
            status_code=404,
            detail=f"No flights found for {origin} → {destination}"
        )
    
    flights = [clean_doc(f) for f in flights]
    
    return {
        "count": len(flights),
        "query": query,
        "flights": flights
    }
```

**What this does:**
- Exact matching search (not keyword or semantic)
- Required: origin and destination
- Optional: airline and date

**Line by line:**
- `Query(...)`: Three dots mean "required parameter"
- `Query(None, ...)`: None means "optional parameter"
- `Optional[str]`: Type hint for optional string
- `origin.upper()`: Convert to uppercase (lhe → LHE)
- Build MongoDB query dictionary
- `{"$regex": airline, "$options": "i"}`: Case-insensitive pattern matching
  - If user searches "pia", it matches "PIA", "Pia", "pia"
- Execute query and return results

**Example usage:**
```
GET /search/route?origin=LHE&destination=JED
GET /search/route?origin=LHE&destination=JED&airline=PIA
GET /search/route?origin=LHE&destination=JED&airline=PIA&date=2026-04-15
```

#### API Endpoint 6: General Search

```python
@app.get("/search")
def search_flights(
    q: str = Query(..., description="Search query"),
    method: str = Query("hybrid", description="Search method: keyword, semantic, or hybrid"),
    limit: int = Query(5, description="Number of results to return")
):
    if method == "keyword":
        results = keyword_search(collection, q, limit=limit)
    elif method == "semantic":
        results = semantic_search(collection, q, limit=limit)
    elif method == "hybrid":
        results = hybrid_search(collection, q, alpha=0.5, limit=limit)
    else:
        raise HTTPException(status_code=400, detail="Invalid search method")
    
    results = [clean_doc(r) for r in results]
    
    return {
        "query": q,
        "method": method,
        "count": len(results),
        "results": results
    }
```

**What this does:**
- Flexible search endpoint
- User can choose search method: keyword, semantic, or hybrid
- Default method is hybrid with alpha=0.5

**Example usage:**
```
GET /search?q=Emirates Dubai
GET /search?q=cheap flights bangkok&method=semantic
GET /search?q=PIA Jeddah&method=keyword&limit=10
```

#### API Endpoint 7: Hybrid Search with Alpha

```python
@app.get("/search/hybrid")
def hybrid_search_with_alpha(
    q: str = Query(..., description="Search query"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Weight for keyword search"),
    limit: int = Query(5, description="Number of results")
):
    results = hybrid_search(collection, q, alpha=alpha, limit=limit)
    results = [clean_doc(r) for r in results]
    
    return {
        "query": q,
        "alpha": alpha,
        "explanation": f"Using {int(alpha*100)}% keyword search and {int((1-alpha)*100)}% semantic search",
        "count": len(results),
        "results": results
    }
```

**What this does:**
- Hybrid search where user can control alpha
- `ge=0.0, le=1.0`: Validation (greater-than-or-equal 0, less-than-or-equal 1)
- Provides explanation of what alpha means

**Example usage:**
```
GET /search/hybrid?q=Dubai flights&alpha=0.5
GET /search/hybrid?q=Dubai flights&alpha=0.8  (more keyword)
GET /search/hybrid?q=Dubai flights&alpha=0.2  (more semantic)
```

#### API Endpoint 8: Optimize Alpha (BONUS)

```python
@app.post("/search/optimize-alpha")
def optimize_alpha_endpoint(
    q: str = Query(..., description="Search query"),
    relevant_ids: List[str] = Query(..., description="List of flight_ids that were relevant")
):
    relevant_docs = list(collection.find({"flight_id": {"$in": relevant_ids}}))
    relevant_mongo_ids = [str(doc['_id']) for doc in relevant_docs]
    
    if not relevant_mongo_ids:
        raise HTTPException(status_code=400, detail="No valid flight IDs found")
    
    optimal_alpha = optimize_alpha(collection, q, relevant_mongo_ids)
    
    results = hybrid_search(collection, q, alpha=optimal_alpha, limit=5)
    results = [clean_doc(r) for r in results]
    
    return {
        "query": q,
        "optimal_alpha": optimal_alpha,
        "explanation": f"Based on your feedback, optimal balance is {int(optimal_alpha*100)}% keyword and {int((1-optimal_alpha)*100)}% semantic",
        "results_with_optimal_alpha": results
    }
```

**What this does:**
- `@app.post`: POST request (not GET)
- User provides query and list of relevant flight IDs
- System finds optimal alpha for this query
- Returns optimal alpha and results using it

**Why POST instead of GET?**
- POST is better for sending lists of data
- GET URLs can get very long with multiple parameters

**Line by line:**
- `List[str]`: Type hint for list of strings
- `{"$in": relevant_ids}`: MongoDB operator to match any value in list
- Convert flight_ids to MongoDB _ids
- Call optimize_alpha function
- Get results with optimal alpha
- Return everything

**Example usage in Postman:**
```
POST /search/optimize-alpha?q=Bangkok flights&relevant_ids=FL1004&relevant_ids=FL1008
```

#### API Endpoint 9: Statistics

```python
@app.get("/stats")
def get_statistics():
    total_flights = collection.count_documents({})
    
    airlines = collection.distinct("airline")
    
    origins = collection.distinct("route.origin")
    destinations = collection.distinct("route.destination")
    
    return {
        "total_flights": total_flights,
        "total_airlines": len(airlines),
        "airlines": airlines,
        "total_origins": len(origins),
        "origins": origins,
        "total_destinations": len(destinations),
        "destinations": destinations
    }
```

**What this does:**
- Provides overview of database
- `count_documents({})`: Count all documents
- `distinct(...)`: Get unique values of a field
- Returns statistics

**Example response:**
```json
{
  "total_flights": 10,
  "total_airlines": 10,
  "airlines": ["PIA", "Emirates", "Turkish Airlines", ...],
  "total_origins": 3,
  "origins": ["LHE", "KHI", "ISB"],
  "total_destinations": 10,
  "destinations": ["JED", "DXB", "IST", ...]
}
```

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**What this does:**
- If we run this file directly: `python main.py`
- Start uvicorn server
- `host="0.0.0.0"`: Accept connections from any IP
- `port=8000`: Run on port 8000
- API will be available at http://localhost:8000

---

## SEARCH IMPLEMENTATION

### 1. Keyword Search

**How it works:**
1. Uses MongoDB's text search feature
2. Searches in origin, destination, and airline fields
3. Returns documents matching any of the search terms
4. Ranks by relevance score

**Advantages:**
- Fast (uses MongoDB index)
- Exact word matching
- Good for specific searches like "PIA" or "Dubai"

**Disadvantages:**
- Can't understand meaning
- "LHE" and "Lahore" won't match
- Typos break search

**When to use:**
- User knows exact airport codes or airline names
- Looking for specific words

### 2. Semantic Search

**How it works:**
1. Convert query to vector (embedding)
2. Compare query vector with document vectors
3. Use cosine similarity to measure closeness
4. Return most similar documents

**Advantages:**
- Understands meaning
- "cheap flights" matches "affordable tickets"
- "Lahore to Dubai" matches "LHE to DXB"
- Handles typos better

**Disadvantages:**
- Slower (compares with all documents)
- Needs embedding model
- Might return unexpected results

**When to use:**
- Natural language queries
- User doesn't know exact codes
- Conversational search

### 3. Hybrid Search

**How it works:**
1. Run both keyword and semantic search
2. Combine scores using weighted formula
3. Adjust balance with alpha parameter

**Formula:**
```
hybrid_score = α × keyword_score + (1-α) × semantic_score
```

**Advantages:**
- Best of both worlds
- Balances precision (keyword) and recall (semantic)
- Flexible (adjust alpha for different queries)

**When to use:**
- Default choice (works well for most queries)
- When you want both exact matching and semantic understanding

### Comparison Example:

**Query:** "affordable flights from Lahore to Bangkok"

**Keyword Search:**
- Looks for: "affordable", "flights", "Lahore", "Bangkok"
- Might miss: LHE→BKK flights (different words)
- Score: Medium

**Semantic Search:**
- Understands: User wants cheap flights LHE→BKK
- Matches: Even if document says "LHE to BKK Thai Airways"
- Score: High

**Hybrid Search (α=0.5):**
- Combines both approaches
- Balances exact matching with semantic understanding
- Score: Highest (gets best of both)

---

## API DESIGN

### RESTful Design Principles

Our API follows REST (Representational State Transfer) principles:

1. **Resource-Based URLs:**
   - `/flights` → Collection of flights
   - `/flights/{id}` → Specific flight
   - `/flights/{id}/prices` → Sub-resource (prices of a flight)

2. **HTTP Methods:**
   - GET: Retrieve data (read-only)
   - POST: Send data (for optimization endpoint)

3. **Status Codes:**
   - 200: Success
   - 404: Not Found
   - 400: Bad Request

4. **JSON Format:**
   - All responses in JSON
   - Easy to parse in any language

### Endpoint Organization

```
/                           → Welcome message
/flights                    → All flights
/flights/{id}              → Single flight
/flights/{id}/prices       → Price history (Part 1)
/search/route              → Exact route search
/search                    → General search
/search/hybrid             → Hybrid search with alpha
/search/optimize-alpha     → Alpha optimization (Bonus)
/stats                     → Statistics
```

### Why This Design?

1. **Logical Grouping:**
   - All search endpoints under `/search`
   - All flight endpoints under `/flights`

2. **Clear Hierarchy:**
   - `/flights/{id}/prices` shows prices belong to a flight

3. **Easy to Extend:**
   - Can add `/flights/{id}/booking` in future
   - Can add `/search/advanced` for complex filters

---

## COMMON VIVA QUESTIONS & ANSWERS

### Q1: Explain your project in 2 minutes.

**Answer:**
"My project is a Flight Price Tracker system. It has two main parts:

Part 1: We track flight ticket prices over time. For example, if a flight is departing on June 1st, 2026, we start tracking its price 6 months before - on December 1st, 2025. We record the price every 15 days, so we get a time-series of prices showing how the ticket price changes over time.

Part 2: We built APIs using FastAPI to access this data. Users can search for flights using three different methods: keyword search (exact matching), semantic search (understanding meaning), and hybrid search (combining both). The hybrid search has an alpha parameter that controls the balance between keyword and semantic search.

We used MongoDB to store the data because it's good for storing arrays (price history) and flexible documents. We implemented all search methods ourselves using sentence-transformers for creating embeddings and calculating cosine similarity."

### Q2: Why MongoDB instead of SQL?

**Answer:**
"I chose MongoDB for several reasons:

1. **Flexible Schema:** Each flight can have different number of price points. In SQL, this would require a separate table with foreign keys, making queries complex. In MongoDB, price_history is just an array in the same document.

2. **Easy to Store Arrays:** Price history is naturally an array. MongoDB stores it directly without normalization.

3. **Good for Time-Series:** MongoDB is optimized for time-series data like our price history.

4. **JSON-like Documents:** Our API returns JSON, and MongoDB stores JSON-like documents (BSON). This makes conversion easy.

5. **Vector Storage:** We store 384-dimensional embeddings for semantic search. MongoDB handles large arrays well.

However, SQL would also work - it's just more complex for this use case."

### Q3: Explain keyword vs semantic search.

**Answer:**
"Keyword search looks for exact words in the documents. For example, if I search 'Dubai Emirates', it finds documents containing these exact words. It uses MongoDB's text index and is very fast. But it can't understand meaning - 'LHE' and 'Lahore' won't match.

Semantic search understands meaning. It converts both the query and documents into vectors (lists of numbers) using a neural network model. Then it compares these vectors using cosine similarity. Documents with similar meaning get high scores, even if they use different words. For example, 'cheap flights' matches 'affordable tickets'.

Keyword is better for specific searches with known terms. Semantic is better for natural language queries. That's why we combine them in hybrid search."

### Q4: What is an embedding?

**Answer:**
"An embedding is a way to represent text as numbers (a vector). For example, 'Lahore to Dubai' becomes [0.123, -0.456, 0.789, ..., 0.234] - a list of 384 numbers.

The key property is: Similar meanings = similar vectors. So 'LHE to DXB' and 'Lahore to Dubai' have similar embeddings even though they use different words.

We create embeddings using a pre-trained model called 'all-MiniLM-L6-v2' from sentence-transformers. This model was trained on millions of sentences to learn which sentences have similar meanings.

We store these embeddings in MongoDB and use cosine similarity to compare them during semantic search."

### Q5: Explain the hybrid search formula.

**Answer:**
"The hybrid search formula is:

```
hybrid_score = α × keyword_score + (1-α) × semantic_score
```

Where α (alpha) is a number between 0 and 1 that controls the balance.

For example, if:
- keyword_score = 0.8
- semantic_score = 0.6
- α = 0.5

Then: hybrid_score = 0.5 × 0.8 + 0.5 × 0.6 = 0.4 + 0.3 = 0.7

If we change α to 0.7 (more keyword weight):
hybrid_score = 0.7 × 0.8 + 0.3 × 0.6 = 0.56 + 0.18 = 0.74

So higher α means we trust keyword search more, lower α means we trust semantic search more. Default is 0.5 which is balanced."

### Q6: What is cosine similarity?

**Answer:**
"Cosine similarity measures how similar two vectors are. It gives a number between 0 and 1:
- 1 means identical (parallel vectors)
- 0 means completely different (perpendicular vectors)

The formula is:
```
similarity = (A · B) / (||A|| × ||B||)
```

Where:
- A · B is dot product (multiply corresponding elements and sum)
- ||A|| is the length of vector A
- ||B|| is the length of vector B

We use this in semantic search to compare the query embedding with document embeddings. Higher similarity means the documents are more relevant to the query."

### Q7: How does alpha optimization work?

**Answer:**
"Alpha optimization finds the best alpha value for a specific query based on user feedback.

Here's how:
1. User does a search and marks which results were relevant
2. We try different alpha values (0.0, 0.1, 0.2, ... , 1.0)
3. For each alpha, we do hybrid search and check how many relevant documents appear in top results
4. We give more weight to relevant documents that appear earlier (position weighting)
5. The alpha that gives the highest score is returned as optimal

For example, if user searches 'Bangkok flights' and marks FL1004 as relevant, we might find that alpha=0.3 puts FL1004 at the top, so 0.3 is returned as optimal alpha for this query."

### Q8: Why FastAPI instead of Flask?

**Answer:**
"FastAPI has several advantages:

1. **Automatic Documentation:** Creates interactive API docs at /docs. I can test all APIs in the browser without Postman.

2. **Type Validation:** Automatically validates request parameters. If user sends invalid data, FastAPI returns a clear error message.

3. **Modern:** Uses Python 3.6+ type hints, which makes code clearer.

4. **Fast:** Built on Starlette and Pydantic, it's faster than Flask.

5. **Easy to Learn:** Similar to Flask but with more features built-in.

Our sir taught us FastAPI, so it was expected in the project."

### Q9: Explain your database schema.

**Answer:**
"Our schema has one collection called 'flights'. Each document has:

1. **flight_id:** Unique identifier (FL1000, FL1001, etc.)
2. **route:** Nested object with origin and destination
3. **airline:** Name of the airline
4. **departure_date:** When the flight departs
5. **base_price:** Starting price
6. **price_history:** Array of price points, each with date, price, and currency
7. **tracking_interval_days:** How often we track (15 days)
8. **embedding:** 384-dimensional vector for semantic search
9. **created_at:** When record was created

This schema is denormalized (everything in one document) which is good for MongoDB because we can retrieve all flight data in one query."

### Q10: What indexes did you create and why?

**Answer:**
"I created five indexes:

1. **route.origin (Ascending):** For searching by origin airport
2. **route.destination (Ascending):** For searching by destination
3. **airline (Ascending):** For filtering by airline
4. **departure_date (Ascending):** For date-based queries
5. **Text Index on origin, destination, airline:** For keyword search

These indexes make queries much faster. Without indexes, MongoDB has to scan all documents (slow). With indexes, MongoDB can directly find relevant documents (fast).

For example, when searching 'route.origin = LHE', MongoDB uses the index to quickly find all flights from Lahore instead of checking every document."

### Q11: How did you generate realistic price data?

**Answer:**
"I used a formula that mimics real flight pricing:

```python
price = base_price × time_factor × random_factor
```

Where:
- **base_price:** Random between $400-$1500 (depends on route distance)
- **time_factor:** Increases as flight date approaches
  - Formula: 1 + (180 - days_until_flight) / 500
  - Far from flight: ≈1.0 (lower price)
  - Near to flight: ≈1.36 (higher price)
- **random_factor:** Random between 0.85 and 1.15 (±15% fluctuation)

This creates realistic patterns where prices generally increase closer to departure but also have random fluctuations."

### Q12: Can you explain one API endpoint in detail?

**Answer:**
"Let me explain the price history endpoint `/flights/{flight_id}/prices`:

This is the main API for Part 1 of the project. When user calls:
```
GET /flights/FL1000/prices
```

The endpoint:
1. Extracts 'FL1000' from the URL
2. Queries MongoDB for this flight_id
3. If not found, returns 404 error
4. If found, returns:
   - Basic flight info (route, airline, date)
   - Full price_history array
   - Current price (last item in array)
   - Lowest price (using min function)
   - Highest price (using max function)

This gives users complete view of how the price changed over time and helps them decide when to book."

---

## KEY TAKEAWAYS FOR VIVA

1. **Be clear about WHY you made each choice** (MongoDB, FastAPI, hybrid search)
2. **Understand the search methods deeply** (keyword, semantic, hybrid)
3. **Know the formulas** (hybrid score, cosine similarity)
4. **Explain line by line if asked** (especially search_utils.py and main.py)
5. **Connect to real-world** ("Just like Google uses hybrid search...")
6. **Be confident** - You built this, you understand it!

---

## FINAL TIPS

### During Viva:

1. **Start with high-level overview** - Don't jump into technical details immediately
2. **Use simple language** - Avoid jargon unless specifically asked
3. **Draw diagrams if allowed** - Visual explanation of hybrid search formula
4. **Give examples** - "For example, when user searches..."
5. **Show confidence** - You understand this project well
6. **Admit if you don't know** - Better than making up answers
7. **Connect to coursework** - "As we learned in class about MongoDB indexes..."
8. **Be prepared for code walkthrough** - They might ask you to explain any function
9. **Know your numbers** - 10 flights, 384-dimensional embeddings, 15-day intervals
10. **Practice explaining** - Explain the project to a friend/family member first

### Common Follow-up Questions:

**"What would you improve?"**
- Add more flights (scale to 1000+)
- Use vector database for faster semantic search
- Add user authentication
- Add real-time price scraping
- Add price prediction using machine learning
- Cache search results for better performance

**"What challenges did you face?"**
- Understanding embeddings initially
- Normalizing scores for hybrid search
- Balancing keyword and semantic search
- Deciding optimal database schema
- Testing different alpha values

**"How would you scale this?"**
- Use MongoDB sharding for large datasets
- Implement caching (Redis)
- Use vector database (Pinecone) for semantic search
- Add load balancer for multiple API servers
- Implement pagination for large result sets

---

## CONCLUSION

This document covers everything you need to know for your viva. Read through it multiple times, practice explaining concepts out loud, and you'll do great!

**Remember:** You built a complete, working system with advanced search capabilities. Be proud of your work and explain it confidently!

Good luck with your viva! 🎓
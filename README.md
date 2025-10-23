# Flight Price Tracker System

**Author:** Abdullah Asif  
**Course:** Advanced Database Systems - 5th Semester BSCS  
**Project Weight:** 25% of grade

## Project Overview

This system tracks flight ticket prices over time and provides APIs to search and retrieve historical pricing data. It supports hybrid search combining keyword and semantic search capabilities.

## Features

### Part 1: Price Tracking System
- Tracks flight prices over defined intervals (15 days)
- Stores time-series price data for multiple routes
- Automated price history generation for 10 different flight routes

### Part 2: Implementation
1. ✅ MongoDB database with sample data for 10 flights
2. ✅ RESTful APIs for flight data retrieval
3. ✅ Hybrid search (keyword + semantic)
4. ✅ BONUS: Alpha optimization based on user feedback

## Technology Stack

- **Language:** Python 3.10
- **Framework:** FastAPI
- **Database:** MongoDB (Community Edition)
- **Search:** Hybrid (Keyword + Semantic using sentence-transformers)
- **Embeddings:** all-MiniLM-L6-v2 model

## Project Structure

```
flight_price_tracker/
│
├── generate_data.py          # Generates realistic flight price data
├── seed_database.py           # Seeds MongoDB with sample data
├── main.py                    # FastAPI application (all APIs)
├── search_utils.py            # Search functions (keyword, semantic, hybrid)
├── requirements.txt           # Python dependencies
├── flights_data.json          # Generated flight data (created after running generate_data.py)
├── README.md                  # This file
└── viva_preparation.md        # Detailed documentation for viva
```

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- MongoDB Community Edition installed and running
- Conda (Anaconda/Miniconda)

### Step 1: Environment Setup

```bash
# Create conda environment
conda create -n flight_tracker python=3.10

# Activate environment
conda activate flight_tracker

# Install dependencies
conda install -c conda-forge fastapi uvicorn pymongo pandas numpy
pip install sentence-transformers python-dateutil
```

### Step 2: Start MongoDB

Make sure MongoDB is running on your system:
```bash
# Check if MongoDB is running
# Open MongoDB Compass and connect to localhost:27017
```

### Step 3: Generate Sample Data

```bash
python generate_data.py
```

This will create `flights_data.json` with sample data for 10 flights.

### Step 4: Seed Database

```bash
python seed_database.py
```

This will:
- Connect to MongoDB
- Create database `flight_tracker_db`
- Create collection `flights`
- Insert 10 flight records with price history
- Generate embeddings for semantic search
- Create necessary indexes

### Step 5: Run API Server

```bash
python main.py
```

Or alternatively:
```bash
uvicorn main:app --reload
```

The API will be available at: `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Interactive API Docs:** http://localhost:8000/docs
- **Alternative Docs:** http://localhost:8000/redoc

### Available Endpoints

#### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

#### 2. Get All Flights
```
GET /flights?limit=10
```
Returns list of all flights with basic information.

#### 3. Get Flight by ID
```
GET /flights/{flight_id}
```
Returns detailed information about a specific flight.

**Example:**
```
GET /flights/FL1000
```

#### 4. Get Price History (Main Part 1 API)
```
GET /flights/{flight_id}/prices
```
Returns time-series price data for a flight.

**Example Response:**
```json
{
  "flight_id": "FL1000",
  "route": "LHE → JED",
  "airline": "PIA",
  "departure_date": "2026-04-15",
  "price_history": [
    {"date": "2025-10-15", "price": 520, "currency": "USD"},
    {"date": "2025-10-30", "price": 545, "currency": "USD"},
    {"date": "2025-11-14", "price": 580, "currency": "USD"}
  ],
  "current_price": {"date": "2025-11-14", "price": 580, "currency": "USD"},
  "lowest_price": {"date": "2025-10-15", "price": 520, "currency": "USD"},
  "highest_price": {"date": "2025-11-14", "price": 580, "currency": "USD"}
}
```

#### 5. Search by Route
```
GET /search/route?origin=LHE&destination=JED&airline=PIA
```
Search flights by exact route matching.

**Parameters:**
- `origin` (required): Origin airport code
- `destination` (required): Destination airport code
- `airline` (optional): Airline name
- `date` (optional): Departure date (YYYY-MM-DD)

#### 6. General Search
```
GET /search?q=Lahore to Dubai&method=hybrid&limit=5
```
Search using keyword, semantic, or hybrid method.

**Parameters:**
- `q` (required): Search query
- `method` (optional): keyword | semantic | hybrid (default: hybrid)
- `limit` (optional): Number of results (default: 5)

#### 7. Hybrid Search with Alpha
```
GET /search/hybrid?q=Emirates flight&alpha=0.5&limit=5
```
Hybrid search with adjustable alpha parameter.

**Parameters:**
- `q` (required): Search query
- `alpha` (optional): 0.0 to 1.0 (default: 0.5)
  - 0.0 = Pure semantic search
  - 0.5 = Balanced
  - 1.0 = Pure keyword search
- `limit` (optional): Number of results

#### 8. Optimize Alpha (BONUS)
```
POST /search/optimize-alpha?q=flights to bangkok&relevant_ids=FL1004&relevant_ids=FL1008
```
Finds optimal alpha value based on user feedback.

**Parameters:**
- `q` (required): Search query
- `relevant_ids` (required): List of flight_ids user found relevant

#### 9. Statistics
```
GET /stats
```
Returns database statistics (total flights, airlines, routes).

## Testing with Postman

1. Import the API into Postman using: `http://localhost:8000/openapi.json`
2. Or manually create requests for each endpoint
3. Example test sequence:

```
1. GET http://localhost:8000/
2. GET http://localhost:8000/flights
3. GET http://localhost:8000/flights/FL1000/prices
4. GET http://localhost:8000/search/route?origin=LHE&destination=JED
5. GET http://localhost:8000/search?q=Emirates Dubai
6. GET http://localhost:8000/search/hybrid?q=Bangkok&alpha=0.3
```

## Sample Routes in Database

1. LHE → JED (PIA)
2. LHE → DXB (Emirates)
3. LHE → IST (Turkish Airlines)
4. KHI → LHR (British Airways)
5. ISB → BKK (Thai Airways)
6. LHE → SIN (Singapore Airlines)
7. KHI → JFK (Qatar Airways)
8. ISB → YYZ (Air Canada)
9. LHE → KUL (Malaysia Airlines)
10. KHI → SYD (Etihad)

## Schema Design

### Flight Document Structure
```json
{
  "flight_id": "FL1000",
  "route": {
    "origin": "LHE",
    "destination": "JED"
  },
  "airline": "PIA",
  "departure_date": "2026-04-15",
  "base_price": 650,
  "price_history": [
    {
      "date": "2025-10-15",
      "price": 520,
      "currency": "USD"
    }
  ],
  "tracking_interval_days": 15,
  "embedding": [0.123, -0.456, ...],
  "created_at": "2025-10-22 14:30:00"
}
```

### Indexes Created
- `route.origin` (Ascending)
- `route.destination` (Ascending)
- `airline` (Ascending)
- `departure_date` (Ascending)
- Text index on origin, destination, airline

## Search Implementation

### 1. Keyword Search
- Uses MongoDB text search
- Searches in origin, destination, and airline fields
- Returns results ranked by text relevance score

### 2. Semantic Search
- Converts query to vector using sentence-transformers
- Calculates cosine similarity with document embeddings
- Returns results ranked by semantic similarity

### 3. Hybrid Search
- Combines keyword and semantic search
- Formula: `hybrid_score = alpha * keyword_score + (1-alpha) * semantic_score`
- Alpha parameter controls the balance

### 4. Alpha Optimization (Bonus)
- Tests different alpha values (0.0 to 1.0)
- Evaluates which alpha gives best results based on user feedback
- Returns optimal alpha for the query

## Troubleshooting

### MongoDB Connection Error
```
Error: Connection refused
Solution: Make sure MongoDB is running. Check MongoDB Compass connection.
```

### Module Not Found Error
```
Error: No module named 'fastapi'
Solution: Activate conda environment and reinstall dependencies
conda activate flight_tracker
pip install -r requirements.txt
```

### Port Already in Use
```
Error: Address already in use
Solution: Use different port
uvicorn main:app --port 8001
```

## Project Submission Checklist

- [x] Source code (all .py files)
- [x] Dataset (flights_data.json)
- [x] README.md
- [x] viva_preparation.md
- [x] requirements.txt
- [ ] Execution screenshots (Postman/Browser)

## Author Information

**Name:** Abdullah Asif  
**Semester:** 5th Semester BSCS  
**Course:** Advanced Database Systems  
**Date:** October 2025

## License

This project is for educational purposes only.
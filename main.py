from fastapi import FastAPI, HTTPException, Query
from pymongo import MongoClient
from typing import Optional, List
from datetime import datetime
import json
from search_utils import keyword_search, semantic_search, hybrid_search, optimize_alpha

# Create FastAPI app
app = FastAPI(
    title="Flight Price Tracker API",
    description="API to track and search flight ticket prices over time",
    version="1.0.0"
)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["flight_tracker_db"]
collection = db["flights"]


# Helper function to clean MongoDB documents for JSON response
def clean_doc(doc):
    """Removes MongoDB _id and embedding fields for cleaner API response"""
    if doc:
        doc['_id'] = str(doc['_id'])
        if 'embedding' in doc:
            del doc['embedding']
    return doc


@app.get("/")
def root():
    """Welcome endpoint - shows API is running"""
    return {
        "message": "Flight Price Tracker API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "get_all_flights": "/flights",
            "get_flight_by_id": "/flights/{flight_id}",
            "get_price_history": "/flights/{flight_id}/prices",
            "search_by_route": "/search/route",
            "search_flights": "/search",
            "hybrid_search": "/search/hybrid"
        }
    }


@app.get("/flights")
def get_all_flights(limit: int = Query(10, description="Number of flights to return")):
    """
    Get all flights with basic information
    """
    flights = list(collection.find({}).limit(limit))
    flights = [clean_doc(f) for f in flights]
    
    return {
        "count": len(flights),
        "flights": flights
    }


@app.get("/flights/{flight_id}")
def get_flight_by_id(flight_id: str):
    """
    Get detailed information about a specific flight
    """
    flight = collection.find_one({"flight_id": flight_id})
    
    if not flight:
        raise HTTPException(status_code=404, detail=f"Flight {flight_id} not found")
    
    return clean_doc(flight)


@app.get("/flights/{flight_id}/prices")
def get_price_history(flight_id: str):
    """
    Get price history for a specific flight
    This is the main API for Part 1 of your project
    
    Input: flight_id
    Output: Time-series of prices
    """
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


@app.get("/search/route")
def search_by_route(
    origin: str = Query(..., description="Origin airport code (e.g., LHE)"),
    destination: str = Query(..., description="Destination airport code (e.g., JED)"),
    airline: Optional[str] = Query(None, description="Airline name (optional)"),
    date: Optional[str] = Query(None, description="Departure date YYYY-MM-DD (optional)")
):
    """
    Search flights by route, airline, and date
    This is for exact matching searches
    """
    # Build query
    query = {
        "route.origin": origin.upper(),
        "route.destination": destination.upper()
    }
    
    if airline:
        query["airline"] = {"$regex": airline, "$options": "i"}  # Case-insensitive search
    
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


@app.get("/search")
def search_flights(
    q: str = Query(..., description="Search query (e.g., 'Lahore to Jeddah', 'Emirates flights')"),
    method: str = Query("hybrid", description="Search method: keyword, semantic, or hybrid"),
    limit: int = Query(5, description="Number of results to return")
):
    """
    General search endpoint - supports keyword, semantic, and hybrid search
    """
    if method == "keyword":
        results = keyword_search(collection, q, limit=limit)
    elif method == "semantic":
        results = semantic_search(collection, q, limit=limit)
    elif method == "hybrid":
        results = hybrid_search(collection, q, alpha=0.5, limit=limit)
    else:
        raise HTTPException(status_code=400, detail="Invalid search method. Use: keyword, semantic, or hybrid")
    
    results = [clean_doc(r) for r in results]
    
    return {
        "query": q,
        "method": method,
        "count": len(results),
        "results": results
    }


@app.get("/search/hybrid")
def hybrid_search_with_alpha(
    q: str = Query(..., description="Search query"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Weight for keyword search (0-1). 0=semantic only, 1=keyword only"),
    limit: int = Query(5, description="Number of results")
):
    """
    Hybrid search with adjustable alpha parameter
    
    Alpha parameter controls the balance:
    - alpha = 0: Pure semantic search
    - alpha = 0.5: Equal balance (default)
    - alpha = 1: Pure keyword search
    """
    results = hybrid_search(collection, q, alpha=alpha, limit=limit)
    results = [clean_doc(r) for r in results]
    
    return {
        "query": q,
        "alpha": alpha,
        "explanation": f"Using {int(alpha*100)}% keyword search and {int((1-alpha)*100)}% semantic search",
        "count": len(results),
        "results": results
    }


@app.post("/search/optimize-alpha")
def optimize_alpha_endpoint(
    q: str = Query(..., description="Search query"),
    relevant_ids: List[str] = Query(..., description="List of flight_ids that were relevant")
):
    """
    BONUS: Optimize alpha value based on user feedback
    
    This endpoint takes a query and list of relevant flight IDs,
    then returns the optimal alpha value for that query
    """
    # Convert flight_ids to MongoDB _ids
    relevant_docs = list(collection.find({"flight_id": {"$in": relevant_ids}}))
    relevant_mongo_ids = [str(doc['_id']) for doc in relevant_docs]
    
    if not relevant_mongo_ids:
        raise HTTPException(status_code=400, detail="No valid flight IDs found")
    
    # Find optimal alpha
    optimal_alpha = optimize_alpha(collection, q, relevant_mongo_ids)
    
    # Get results with optimal alpha
    results = hybrid_search(collection, q, alpha=optimal_alpha, limit=5)
    results = [clean_doc(r) for r in results]
    
    return {
        "query": q,
        "optimal_alpha": optimal_alpha,
        "explanation": f"Based on your feedback, optimal balance is {int(optimal_alpha*100)}% keyword and {int((1-optimal_alpha)*100)}% semantic",
        "results_with_optimal_alpha": results
    }


@app.get("/stats")
def get_statistics():
    """
    Get database statistics
    """
    total_flights = collection.count_documents({})
    
    # Get unique airlines
    airlines = collection.distinct("airline")
    
    # Get unique routes
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


# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
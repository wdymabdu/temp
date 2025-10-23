from pymongo import MongoClient, ASCENDING, TEXT
from sentence_transformers import SentenceTransformer
import json

# Simple function to connect to MongoDB
def get_database():
    """
    Connects to local MongoDB and returns database
    """
    # Connect to MongoDB (default local connection)
    client = MongoClient("mongodb://localhost:27017/")
    
    # Create/get database named 'flight_tracker_db'
    db = client["flight_tracker_db"]
    
    return db


def create_embeddings_for_flights(flights):
    """
    Creates vector embeddings for semantic search
    Embedding is created from: origin, destination, and airline
    """
    print("Loading embedding model...")
    # Load a small, fast model for creating embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for flight in flights:
        # Create a text description of the flight
        text = f"{flight['route']['origin']} to {flight['route']['destination']} {flight['airline']}"
        
        # Generate embedding (converts text to numbers/vector)
        embedding = model.encode(text).tolist()
        
        # Add embedding to flight document
        flight['embedding'] = embedding
    
    return flights


def seed_database():
    """
    Main function to load data into MongoDB
    """
    print("Connecting to MongoDB...")
    db = get_database()
    
    # Get the flights collection (like a table in SQL)
    collection = db["flights"]
    
    # Clear existing data
    collection.delete_many({})
    print("Cleared existing data")
    
    # Load flight data from JSON file
    print("Loading flight data from JSON...")
    with open("flights_data.json", "r") as f:
        flights = json.load(f)
    
    # Create embeddings for semantic search
    flights = create_embeddings_for_flights(flights)
    
    # Insert all flights into MongoDB
    print("Inserting data into MongoDB...")
    result = collection.insert_many(flights)
    print(f"Inserted {len(result.inserted_ids)} flight records")
    
    # Create indexes for better search performance
    print("Creating indexes...")
    
    # Index for route search
    collection.create_index([("route.origin", ASCENDING)])
    collection.create_index([("route.destination", ASCENDING)])
    
    # Index for airline search
    collection.create_index([("airline", ASCENDING)])
    
    # Index for date search
    collection.create_index([("departure_date", ASCENDING)])
    
    # Text index for keyword search
    collection.create_index([
        ("route.origin", TEXT),
        ("route.destination", TEXT),
        ("airline", TEXT)
    ])
    
    print("Database seeding completed successfully!")
    
    # Print sample document
    print("\nSample document from database:")
    sample = collection.find_one()
    # Remove embedding for cleaner display (it's very long)
    if sample and 'embedding' in sample:
        sample['embedding'] = f"[{len(sample['embedding'])} dimensional vector]"
    print(json.dumps(sample, indent=2, default=str))


if __name__ == "__main__":
    seed_database()
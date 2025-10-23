import random
from datetime import datetime, timedelta
import json

# Simple function to generate realistic flight price data
def generate_flight_prices():
    """
    Generates sample flight data for 10 different routes
    Each flight has price tracked over 6 months before departure
    """
    
    # Define 10 different flight routes
    routes = [
        {"from": "LHE", "to": "JED", "airline": "PIA"},
        {"from": "LHE", "to": "DXB", "airline": "Emirates"},
        {"from": "LHE", "to": "IST", "airline": "Turkish Airlines"},
        {"from": "KHI", "to": "LHR", "airline": "British Airways"},
        {"from": "ISB", "to": "BKK", "airline": "Thai Airways"},
        {"from": "LHE", "to": "SIN", "airline": "Singapore Airlines"},
        {"from": "KHI", "to": "JFK", "airline": "Qatar Airways"},
        {"from": "ISB", "to": "YYZ", "airline": "Air Canada"},
        {"from": "LHE", "to": "KUL", "airline": "Malaysia Airlines"},
        {"from": "KHI", "to": "SYD", "airline": "Etihad"}
    ]
    
    all_flights_data = []
    
    # Generate data for each route
    for idx, route in enumerate(routes):
        # Flight departure date (randomly between Jan 2026 to Jun 2026)
        days_ahead = random.randint(90, 270)  # 3 to 9 months ahead
        flight_date = datetime.now() + timedelta(days=days_ahead)
        
        # Base price depends on route distance (simulated)
        base_price = random.randint(400, 1500)
        
        # Track prices starting 6 months before flight
        tracking_start_date = flight_date - timedelta(days=180)
        
        # Generate price history (every 15 days)
        price_history = []
        current_date = tracking_start_date
        
        while current_date <= datetime.now():
            # Price fluctuates around base price
            # Closer to flight date, price generally increases
            days_until_flight = (flight_date - current_date).days
            
            # Price increase factor (prices go up as flight date approaches)
            time_factor = 1 + (180 - days_until_flight) / 500
            
            # Random fluctuation
            random_factor = random.uniform(0.85, 1.15)
            
            # Calculate final price
            price = int(base_price * time_factor * random_factor)
            
            price_history.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "price": price,
                "currency": "USD"
            })
            
            # Move to next tracking date (15 days later)
            current_date += timedelta(days=15)
        
        # Create flight document
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
    
    return all_flights_data


# Main execution
if __name__ == "__main__":
    print("Generating flight price data...")
    
    flights = generate_flight_prices()
    
    # Save to JSON file
    with open("flights_data.json", "w") as f:
        json.dump(flights, f, indent=2)
    
    print(f"Generated {len(flights)} flights data")
    print("Data saved to flights_data.json")
    print("\nSample flight:")
    print(json.dumps(flights[0], indent=2))
from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime
import pytz

app = Flask(__name__)

# MongoDB Atlas connection string
mongo_uri = "$MONGO_URI"

# Connect to MongoDB Atlas
try:
    client = MongoClient(mongo_uri)
    db = client["waste_management"]
    collection = db["waste_data"]
    print("Connected to MongoDB Atlas")
except Exception as error:
    print(f"Error connecting to MongoDB: {error}")


def get_day_of_week(date_string):
    """Get day of week from date string"""
    days_of_week = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    date_obj = datetime.strptime(date_string, "%Y-%m-%d")
    return days_of_week[date_obj.weekday()]


def get_current_datetime_ist():
    """Get current date and time in IST timezone"""
    # Get current UTC time
    utc_now = datetime.utcnow()

    # Convert to IST timezone
    ist_timezone = pytz.timezone("Asia/Kolkata")
    ist_time = utc_now.replace(tzinfo=pytz.utc).astimezone(ist_timezone)

    # Format date and time
    date = ist_time.strftime("%Y-%m-%d")
    time = ist_time.strftime("%H:%M:%S")

    # Get day of week
    day_of_week = get_day_of_week(date)

    return {"date": date, "time": time, "day_of_week": day_of_week}


@app.route("/api/waste-data", methods=["POST"])
def receive_waste_data():
    """Endpoint to receive waste data"""
    try:
        # Get the current date and time in IST
        datetime_info = get_current_datetime_ist()

        # Get request data
        request_data = request.get_json()
        if not request_data:
            return "No JSON data received", 400

        # Add date, time, and day_of_week to the request data
        new_waste_data = {
            **request_data,
            "date": datetime_info["date"],
            "time": datetime_info["time"],
            "day_of_week": datetime_info["day_of_week"],
        }

        # Find the document with the highest entry_id in the collection
        highest_entry = collection.find_one(sort=[("entry_ID", -1)])
        next_entry_id = highest_entry["entry_ID"] + 1 if highest_entry else 1

        # Add the new incremented entry_id to the data
        new_waste_data["entry_ID"] = next_entry_id

        # Insert the new document to the MongoDB collection
        result = collection.insert_one(new_waste_data)

        print("New waste data saved:", new_waste_data)
        return "Data received and saved successfully", 201

    except Exception as error:
        print(f"Error saving waste data: {error}")
        return "Error saving waste data", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

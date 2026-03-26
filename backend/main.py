from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from datetime import datetime
import sys
import os

from database import init_db, insert_log

# Add model directory to path
model_dir = os.path.join(os.path.dirname(__file__), "../model")
sys.path.insert(0, model_dir)
from inference import predict_image

app = FastAPI()

init_db()

def get_recommendations(coffee_class):
    recommendations = {
        "Dark": {
            "drinks": ["Espresso", "Black Coffee", "Cold Brew"],
            "description": "Dark roasts are bold and intense, perfect for espresso-based drinks that can handle strong flavors."
        },
        "Medium": {
            "drinks": ["Cappuccino", "Latte", "Americano", "Drip Coffee"],
            "description": "Medium roasts offer balanced flavor, ideal for creamy drinks like cappuccino and latte."
        },
        "Light": {
            "drinks": ["Pour-over", "French Press", "Cold Brew", "Iced Coffee"],
            "description": "Light roasts preserve more of the bean's original flavors, great for brewing methods that highlight subtle notes."
        },
        "Green": {
            "drinks": ["Roast First"],
            "description": "Green beans need to be roasted before brewing. Consider medium or light roast for best results."
        }
    }
    return recommendations.get(coffee_class, {"drinks": ["General Coffee"], "description": "Standard coffee brewing methods apply."})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = predict_image(image)

    # Add recommendations
    recommendations = get_recommendations(result["class"])
    result["recommendations"] = recommendations

    # Log to DB
    insert_log(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        result["class"],
        result["confidence"],
        "upload"
    )

    return result

@app.get("/logs")
def get_logs():
    import sqlite3
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    db_path = os.path.join(data_dir, "logs.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT * FROM predictions")
    rows = c.fetchall()

    conn.close()
    return rows
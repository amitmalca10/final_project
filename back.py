from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import bcrypt
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import csv  # CSV processing library
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from starlette.responses import JSONResponse
from classification import main_train
from classification import find_data_per_smell
from classification import data_for_bar_chart
from classification import count_unique
_name_="_main_"
class smell:
    def __init__(self, name:str,data:list):
        self._name = name  # _name is a protected attribute
        self._data = data
    def set_name(self, name):
        self._name = name


        # Getter for name
    def get_name(self):
        return self._name

    def set_data(self, data):
        self._data = data

        # Getter for name

    def get_data(self):
        return self._data


last_smell_ditected=smell("",[])


uri = "mongodb+srv://lielsgraphic:cTHXbaYw7i82m5wS@smelltify.dsin9.mongodb.net/?retryWrites=true&w=majority&appName=Smelltify"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['Smelltify']
collection = db['Users']


# Define the FastAPI app
app = FastAPI()
origins=["http://localhost:5173", "http://localhost:5174"]
app.add_middleware(
    CORSMiddleware,
    allow_origins="http://localhost:5173",
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"]   # Allow all headers
)


# Define a directory where files will be saved
UPLOAD_DIRECTORY = "uploaded_files"

# Ensure the directory exists
Path(UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)

csv_data_list = []
# Define Pydantic models for the request body
class features(BaseModel):
    feature_name: str
class User(BaseModel):
    username: str
    password: str
    email: str

class User_Redister(BaseModel):
    username: str
    password: str

# Model for smell data
class SmellData(BaseModel):
    smell_name: str
    fragrance_types: dict

labels = ['alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal', 'anisic',
          'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy', 'bergamot',
          'berry', 'bitter', 'black currant', 'brandy', 'burnt', 'buttery', 'cabbage',
          'camphoreous', 'caramellic', 'cedar', 'celery', 'chamomile', 'cheesy',
          'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean', 'clove', 'cocoa',
          'coconut', 'coffee', 'cognac', 'cooked', 'cooling', 'cortex', 'coumarinic',
          'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 'fatty',
          'fermented', 'fishy', 'floral', 'fresh', 'fruit skin', 'fruity', 'garlic',
          'gassy', 'geranium', 'grape', 'grapefruit', 'grassy', 'green', 'hawthorn',
          'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth', 'jasmin', 'juicy',
          'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery', 'lemon', 'lily',
          'malty', 'meaty', 'medicinal', 'melon', 'metallic', 'milky', 'mint', 'muguet',
          'mushroom', 'musk', 'musty', 'natural', 'nutty', 'odorless', 'oily', 'onion',
          'orange', 'orangeflower', 'orris', 'ozone', 'peach', 'pear', 'phenolic',
          'pine', 'pineapple', 'plum', 'popcorn', 'potato', 'powdery', 'pungent',
          'radish', 'raspberry', 'ripe', 'roasted', 'rose', 'rummy', 'sandalwood',
          'savory', 'sharp', 'smoky', 'soapy', 'solvent', 'sour', 'spicy', 'strawberry',
          'sulfurous', 'sweaty', 'sweet', 'tea', 'terpenic', 'tobacco', 'tomato',
          'tropical', 'vanilla', 'vegetable', 'vetiver', 'violet', 'warm', 'waxy',
          'weedy', 'winey', 'woody']

# Function to hash passwords
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')
@app.get("/Users")
async def Users():
    return "check"


# Sign-up endpoint
@app.post("/sign-up")
async def sign_up(user: User):
    # Check if the username already exists in the database
    existing_username = collection.find_one({"username": user.username})
    existing_mail = collection.find_one({"email": user.email})
    if existing_username or existing_mail :
        raise HTTPException(status_code=400, detail="Username or Email already exists")

    # Hash the password
    hashed_password = hash_password(user.password)

    # Insert the new user into the MongoDB database
    new_user = {
        "username": user.username,
        "password": hashed_password,
        "email": user.email
    }
    collection.insert_one(new_user)



    return {"message": "User created successfully"}

# Sign-up endpoint
@app.post("/sign-in")
async def sign_in(user: User_Redister):
    # Check if the username exists in the database
    existing_user = collection.find_one({"username": user.username})
    if not existing_user:
        raise HTTPException(status_code=400, detail="User not found")

    # Get the stored hashed password
    stored_hashed_password = existing_user.get("password")

    # Verify if the entered password matches the stored hashed password
    if not bcrypt.checkpw(user.password.encode('utf-8'), stored_hashed_password.encode('utf-8')):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"message": "Login successful"}


# CSV upload endpoint
@app.post("/upload-csv")
async def upload_csv(file: UploadFile):

    try:
        print(f"Received file: {file.filename}")
        content = await file.read()
        csv_content = content.decode('utf-8').splitlines()
        csv_reader = csv.reader(csv_content)
        data = [row for row in csv_reader]
        for row in data:
            if (len(row)!=138):
                return JSONResponse(content={
                    'messege': 'no'
                })  # send the data to the front
            for feature in row:
                if (int(feature)!=0 and int(feature)!=1):
                    return JSONResponse(content={
                        'messege': 'no_binar'
                    })  # send the data to the front

        print(f"Running main_train with data: {data[:2]}")
        response = main_train(data)
        print(f"Prediction response: {response}")

        ingredients = response[0].split(";")
        print(f"Ingredients: {ingredients}")
        last_smell_ditected.set_data(data[0])
        last_smell_ditected.set_name(response[0])
        print(last_smell_ditected.get_name())
        return JSONResponse(content={
                'messege': "pass"
        })  # send the data to the front

    except Exception as e:
        print(f"Error during CSV upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


# Endpoint to retrieve the saved CSV data from memory
@app.get("/get-csv-data")
async def get_csv_data():
    if not csv_data_list:
        return {"message": "No CSV data found."}
    return {"csv_data": csv_data_list}



@app.get("/get_smell")
async def get_smell_deatils():


    if (last_smell_ditected.get_name()==""):#Check if got name in the last page
        raise HTTPException(status_code=400, detail="smell not found")
    data_for_bar=data_for_bar_chart(last_smell_ditected.get_data(),last_smell_ditected.get_name())#gets the data for the bar chart
    data_for_unique=count_unique(last_smell_ditected.get_name())#counts the unique smells create this smell
    return JSONResponse(content={
        'smell_name': str(last_smell_ditected.get_name()),
        'smell_data': data_for_bar,
        'unique': str(data_for_unique)
    })#send the data to the front


#read the data for pie chart
df = pd.read_csv('data_for_proj.csv')
# Convert the Dataframe to a 2D array
data_for_pie= df.values.tolist()
@app.get('/get_pie_details')
async def get_pie_details():

   my_data=[]
   for label in labels:
       my_data.append(find_data_per_smell(label,data_for_pie))#returns the data for pie chart
   print(my_data)
   return JSONResponse(content={
        'chart_title': 'Default Title',
        'chart_data': my_data  # Example data
    })



# Run the FastAPI server using uvicorn
if _name_ == "_main_":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


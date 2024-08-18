import google.generativeai as genai
from dotenv import load_dotenv
import os

def generate_recipe(dish):
    load_dotenv()
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([f"Generate a recipe for {dish}"])
    print(response)
    candidate = response._result.candidates[0]
    recipe = candidate.content.parts[0].text
    print(recipe)
    
    return recipe



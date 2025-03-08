import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

client = Groq(api_key=GROQ_API_KEY)

def generate_recommendation_explanation(student_data, predicted_major_name, neighbor_summary):
   
    marks = {
        "Marks in Math": student_data[0],
        "Marks in Physics": student_data[1],
        "Marks in Chemistry": student_data[2],
        "Marks in Art": student_data[3],
        "Marks in Economics": student_data[4],
        "Preference": student_data[5],
    }

    preference_text = "Science" if marks["Preference"] == 1 else "Other"

    prompt = f"""
    You are an AI assistant helping with academic guidance. A KNN model has predicted a recommended major
    for a student based on their academic performance and preferences. Here's the input data for the student:
    
    - Marks in Math: {marks['Marks in Math']}
    - Marks in Physics: {marks['Marks in Physics']}
    - Marks in Chemistry: {marks['Marks in Chemistry']}
    - Marks in Art: {marks['Marks in Art']}
    - Marks in Economics: {marks['Marks in Economics']}
    - Preference: {marks['Preference']} ({preference_text})
    
    The KNN model recommended: {predicted_major_name}.
    
    Context:
    The recommendation is based on the nearest neighbors of the student in the training data. These neighbors include:
    {neighbor_summary}.
    
    Please provide a detailed explanation of why this recommendation makes sense based on the input marks and context. 
    If relevant, include the rationale for why the student's scores and preferences align with this major.
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an academic advisor with expertise in student guidance."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content

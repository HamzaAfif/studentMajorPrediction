import pandas as pd
import random

def generate_enhanced_dataset(num_students=500):
    data = []
    majors = ["Engineering", "Design", "Science", "Business", "Arts"]
    preferences = ["Science", "Art", "Engineering", "Business"]

    for _ in range(num_students):

        major = random.choice(majors)

        if major == "Engineering":
            marks_math = random.randint(85, 100)
            marks_physics = random.randint(80, 100)
            marks_chemistry = random.randint(75, 95)
            marks_art = random.randint(50, 70)
            marks_economics = random.randint(60, 80)
            preference = random.choice(["Engineering", "Science"])
        elif major == "Science":
            marks_math = random.randint(75, 95)
            marks_physics = random.randint(85, 100)
            marks_chemistry = random.randint(80, 100)
            marks_art = random.randint(50, 70)
            marks_economics = random.randint(60, 80)
            preference = random.choice(["Science", "Engineering"])
        elif major == "Design":
            marks_math = random.randint(50, 70)
            marks_physics = random.randint(60, 75)
            marks_chemistry = random.randint(50, 70)
            marks_art = random.randint(85, 100)
            marks_economics = random.randint(60, 75)
            preference = random.choice(["Art", "Science"])
        elif major == "Business":
            marks_math = random.randint(70, 85)
            marks_physics = random.randint(60, 80)
            marks_chemistry = random.randint(60, 80)
            marks_art = random.randint(60, 80)
            marks_economics = random.randint(85, 100)
            preference = random.choice(["Business", "Engineering", "Science"])
        elif major == "Arts":
            marks_math = random.randint(60, 75)
            marks_physics = random.randint(50, 70)
            marks_chemistry = random.randint(50, 70)
            marks_art = random.randint(85, 100)
            marks_economics = random.randint(60, 75)
            preference = "Art"
      
        if random.random() > 0.85:  
            preference = random.choice(preferences)
        
        data.append([marks_math, marks_physics, marks_chemistry, marks_art, marks_economics, preference, major])
    
    return pd.DataFrame(data, columns=["Marks_Math", "Marks_Physics", "Marks_Chemistry", "Marks_Art", "Marks_Economics", "Preferences", "Recommended_Major"])

df = generate_enhanced_dataset(num_students=1000)  

df.to_csv('student_data.csv', index=False)
print("Enhanced dataset saved as 'enhanced_student_data.csv'.")

print(df.head())

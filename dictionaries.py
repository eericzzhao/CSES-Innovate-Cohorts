# Dictionaries have a key:value structure
student = {"name": "Eric", "year": 4}

# We can also add additional key:value pairs by...
student["gpa"] = 3.9

# We can also print out every key-value pair by...
for k,v in student.items():
    print(k,v)
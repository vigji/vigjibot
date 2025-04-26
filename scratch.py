from curses.ascii import isdigit
from pathlib import Path
import json


starting_file = Path("/Users/vigji/Desktop/all_models.txt")


with open(starting_file, "r") as f:
    docs = f.readlines()

all_models = []

for index, line in enumerate(docs):
    model_dict = {}
    if line.split(".")[0].isdigit():
        model_n = int(line.split(".")[0])
        model_name = line.split(".")[1].strip().lower().replace(" ", "-")
        full_model_name = f"{model_n:02d}-{model_name}"
        model_dict["model_name"] = full_model_name
        model_dict["description"] = docs[index + 1].strip()
        all_models.append(model_dict)


with open("all_models.json", "w") as f:
    json.dump(all_models, f, indent=4)

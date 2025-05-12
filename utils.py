import json


def load_forecasters_dict() -> dict:
    with open("all_models.json", "r") as f:
        all_models = json.load(f)
    forecasters_dict = {}
    for model in all_models:
        forecasters_dict[model["model_name"]] = model["description"]

    return forecasters_dict

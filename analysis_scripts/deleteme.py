from mootlib import MootlibMatcher
from dotenv import load_dotenv
import os
import pandas as pd
# load_dotenv()

matcher = MootlibMatcher()
test_question = "US government shutdown in 2025?"

response = pd.DataFrame(matcher.find_similar_questions(test_question, n_results=10, min_similarity=0.))

def _format_questions_and_answers(response):
    final_response = ["\nYou can condition your forecasts on the following questions and answers, if you find them relevant (they are not necessarily related to the question you are forecasting on):\n"]

    for question, answer, source in zip(response["question"], 
                                                  response["formatted_outcomes"], 
                                                  response["source_platform"]):
                                                  # response["similarity_score"]):
        final_response.append(f"{question}: {answer} ({source})") # {similarity_score})")

    return "\n".join(final_response)


print(_format_questions_and_answers(response))



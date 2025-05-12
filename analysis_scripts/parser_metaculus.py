from datetime import datetime, timedelta
import asyncio
from re import I
import pandas as pd

from forecasting_tools import MetaculusApi, ApiFilter, MetaculusQuestion

start_date = datetime(2024, 10, 1)
one_year_from_now = datetime.now() + timedelta(days=365)

DEFAULT_FILTER = ApiFilter(
    allowed_statuses=["open"],
    allowed_types=["binary"],
    num_forecasters_gte=40,
    scheduled_resolve_time_lt=one_year_from_now,
    includes_bots_in_aggregates=False,
    community_prediction_exists=True,
    # publish_time_gt=start_date,
)

class MyMetaculusApi(MetaculusApi):
    @classmethod
    async def grab_all_questions_with_filter(
        cls, filter: ApiFilter = None
    ) -> list[MetaculusQuestion]:

        # This is reachable - the filter parameter is optional and can be None
        if filter is None:
            filter = DEFAULT_FILTER

        questions: list[MetaculusQuestion] = []
        more_questions_available = True
        page_num = 0
        while more_questions_available:
            offset = page_num * cls.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
            new_questions, continue_searching = (
                cls._grab_filtered_questions_with_offset(filter, offset)
            )
            questions.extend(new_questions)
            if not continue_searching:
                more_questions_available = False
            page_num += 1
            await asyncio.sleep(0.1)
        return questions
    
    @classmethod
    def get_all_questions_df(self):
        questions = asyncio.run(MyMetaculusApi.grab_all_questions_with_filter())
        community_predictions = [
            rep.community_prediction_at_access_time for rep in questions
        ]
        num_forecasters = [
            rep.num_forecasters for rep in questions
        ]
        question_ids = [rep.id_of_question for rep in questions]
        question_texts = [rep.question_text for rep in questions]
        question_urls = [rep.page_url for rep in questions]
        publication_times = [rep.published_time for rep in questions]

        print(len(community_predictions), len(num_forecasters), len(question_ids), len(question_texts), len(question_urls), len(publication_times))
        model_df = pd.DataFrame(
            {
                "outcome_probabilities": [[p, 1-p] for p in community_predictions],
                "outcomes": [["yes", "no"] for _ in range(len(community_predictions))],
                "formatted_outcomes": [f"Yes {p:.2f}; No {1-p:.2f}" for p in community_predictions],
                "n_forecasters": num_forecasters,
                "id": ["metaculus_" + str(q_id) for q_id in question_ids],
                "question": question_texts,
                "url": question_urls,
                "published_time": publication_times,
                "source_platform": "Metaculus",
            }
        )
        return model_df

if __name__ == "__main__":
    questions = asyncio.run(MyMetaculusApi.get_all_questions_df())
    print(len(questions))

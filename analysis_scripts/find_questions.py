# %%
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import typeguard

from forecasting_tools import MetaculusApi, ApiFilter, MetaculusQuestion

class MyMetaculusApi(MetaculusApi):
    @classmethod
    async def grab_all_questions_with_filter(
        cls, filter: ApiFilter
    ) -> list[MetaculusQuestion]:

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

# %%
start_date = datetime(2024, 10, 1)
one_year_from_now = datetime.now() + timedelta(days=365)

api_filter = ApiFilter(
            allowed_statuses=["open"],
            allowed_types=["binary"],
            num_forecasters_gte=40,
            scheduled_resolve_time_lt=one_year_from_now,
            includes_bots_in_aggregates=False,
            community_prediction_exists=True,
            # publish_time_gt=start_date,
        )


api_filter = ApiFilter(
            allowed_statuses=["open"],
            allowed_types=["binary"],
            num_forecasters_gte=40,
            scheduled_resolve_time_lt=one_year_from_now,
            includes_bots_in_aggregates=False,
            community_prediction_exists=True,
        )
n_questions = MetaculusApi._determine_how_many_questions_match_filter(api_filter)
# number_of_questions = 1000
# %%
questions = await MyMetaculusApi.grab_all_questions_with_filter(
            api_filter,
            # num_questions=n_questions,
            # randomly_sample=False,
        )
print(len(questions))

# %%
num_of_questions_to_return = 200
questions = asyncio.run(
            MetaculusApi.get_questions_matching_filter(
                api_filter,
                num_questions=num_of_questions_to_return,
                randomly_sample=True,
            )
        )
questions = typeguard.check_type(questions, list)
# %%
questions = MetaculusApi.get_benchmark_questions(num_of_questions_to_return=200)
# %%

n_questions






# %%

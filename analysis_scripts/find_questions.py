# %%
import pandas as pd
from datetime import datetime, timedelta

from forecasting_tools import MetaculusApi, ApiFilter

start_date = datetime(2025, 1, 1)
one_year_from_now = datetime.now() + timedelta(days=365)

api_filter = ApiFilter(
            # allowed_statuses=["open"],
            allowed_types=["binary"],
            num_forecasters_gte=40,
            scheduled_resolve_time_lt=one_year_from_now,
            includes_bots_in_aggregates=False,
            community_prediction_exists=True,
            # publish_time_gt=start_date,
        )


number_of_questions = 1000
questions = await MetaculusApi.get_questions_matching_filter(
    api_filter,
    num_questions=number_of_questions,
    # randomly_sample=True,
)

# %%

len(questions)







# %%

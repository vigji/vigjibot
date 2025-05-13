#!/usr/bin/env python3
"""
Test that CustomForecaster.run_research correctly caches results across instances.
"""

import asyncio
import os

from custom_forecaster import CustomForecaster
from forecasting_tools import MetaculusQuestion


class DummyQuestion(MetaculusQuestion):
    def __init__(self, page_url: str, question_text: str) -> None:
        self.page_url = page_url
        self.question_text = question_text
        self.background_info = ""
        self.resolution_criteria = ""
        self.fine_print = ""
        self.unit_of_measure = None
        self.open_upper_bound = True
        self.open_lower_bound = True
        self.upper_bound = None
        self.lower_bound = None


async def test_cache() -> None:
    # Ensure no real API calls fall back to empty string research
    # First fetch should miss cache
    question = DummyQuestion("http://example.com/q1", "Will X happen?")
    bot1 = CustomForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        forecaster_description="",
        forecaster_name="",
        llms={"default": "openrouter/openai/o4-mini"},
    )
    print("Fetching research first time (expected miss)...")
    result1 = await bot1.run_research(question)

    # Second fetch on a new instance should hit cache
    bot2 = CustomForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        forecaster_description="",
        forecaster_name="",
        llms={"default": "openrouter/openai/o4-mini"},
    )
    print("Fetching research second time (expected hit)...")
    result2 = await bot2.run_research(question)

    print(f"Result1 == Result2: {result1 == result2}")
    print(f"Research output: {result1!r}")


if __name__ == "__main__":
    asyncio.run(test_cache())

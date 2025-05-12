from datetime import datetime, timedelta
import asyncio
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Any

from forecasting_tools import MetaculusApi, ApiFilter, MetaculusQuestion
from common_markets import PooledMarket, BaseMarket, BaseScraper

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


@dataclass
class MetaculusMarket(BaseMarket):
    id: str
    question: str
    outcomes: List[str]
    outcome_probabilities: List[float]
    formatted_outcomes: str
    url: str
    published_time: datetime
    n_forecasters: int
    raw_question: MetaculusQuestion

    @classmethod
    def from_metaculus_question(cls, question: MetaculusQuestion) -> "MetaculusMarket":
        prob = question.community_prediction_at_access_time
        if prob is None:
            prob = 0.5

        return cls(
            id=f"metaculus_{question.id_of_question}",
            question=question.question_text,
            outcomes=["yes", "no"],
            outcome_probabilities=[prob, 1 - prob],
            formatted_outcomes=f"Yes {prob:.2f}; No {1-prob:.2f}",
            url=question.page_url,
            published_time=question.published_time,
            n_forecasters=question.num_forecasters,
            raw_question=question,
        )

    def to_pooled_market(self) -> PooledMarket:
        return PooledMarket(
            id=self.id,
            question=self.question,
            outcomes=self.outcomes,
            outcome_probabilities=self.outcome_probabilities,
            formatted_outcomes=self.formatted_outcomes,
            url=self.url,
            published_at=self.published_time,
            source_platform="Metaculus",
            volume=None,  # Not available in Metaculus
            n_forecasters=self.n_forecasters,
            comments_count=None,  # Not directly available
            original_market_type="BINARY",  # Currently only handling binary questions
            is_resolved=None,  # Would need additional logic to determine
            raw_market_data=self.raw_question,
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
            (
                new_questions,
                continue_searching,
            ) = cls._grab_filtered_questions_with_offset(filter, offset)
            questions.extend(new_questions)
            if not continue_searching:
                more_questions_available = False
            page_num += 1
            await asyncio.sleep(0.1)
        return questions


class MetaculusScraper(BaseScraper):
    def __init__(self, filter: Optional[ApiFilter] = None):
        self.filter = filter or DEFAULT_FILTER
        self.api = MyMetaculusApi  # Use the existing MyMetaculusApi class

    async def __aenter__(self):
        """Async context manager entry - nothing to initialize for Metaculus."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - nothing to clean up for Metaculus."""
        pass

    async def fetch_markets(
        self, only_open: bool = True, **kwargs: Any
    ) -> List[MetaculusMarket]:
        """
        Fetch markets from Metaculus using the existing MyMetaculusApi logic.

        Args:
            only_open: If True, only fetch open markets (handled via filter)
            **kwargs: Additional parameters (unused for Metaculus)

        Returns:
            List of MetaculusMarket objects
        """
        if only_open and "allowed_statuses" not in self.filter.__dict__:
            self.filter.allowed_statuses = ["open"]

        questions = await self.api.grab_all_questions_with_filter(self.filter)
        markets = [MetaculusMarket.from_metaculus_question(q) for q in questions]
        return markets


async def main():
    print("Starting MetaculusScraper example...")
    start_time = time.time()

    async with MetaculusScraper() as scraper:
        markets = await scraper.get_pooled_markets(only_open=True)

        end_time = time.time()
        print(f"Fetching took {end_time - start_time:.2f} seconds.")
        print(f"Fetched {len(markets)} Metaculus markets.")

        if markets:
            print("\nDetails of the first pooled market:")
            pprint(markets[0].__dict__)

            df_pooled = pd.DataFrame([pm.__dict__ for pm in markets])
            print(f"\nCreated DataFrame with {len(df_pooled)} pooled markets.")
        else:
            print("No markets were fetched from Metaculus.")


if __name__ == "__main__":
    import time
    from pprint import pprint

    asyncio.run(main())

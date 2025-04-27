from main import TemplateForecaster
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar, Union
from utils import load_forecasters_dict
import asyncio

from forecasting_tools import (
    MetaculusQuestion,
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    clean_indents,
)

logger = logging.getLogger(__name__)
T = TypeVar('T', float, PredictedOptionList, NumericDistribution)

forecasters_dict = load_forecasters_dict()

model_name = "18-paranoid-conspiracy-minded-forecaster"
forecaster_description = forecasters_dict[model_name]
class CustomForecaster(TemplateForecaster):

    _max_concurrent_questions = 5  # Set this to whatever works for your search-provider/ai-model rate limits
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # Cache research results per question URL across instances
    _research_cache: dict[str, str] = {}

    # Locks per question URL to avoid concurrent fetches of the same key
    _research_locks: dict[str, asyncio.Lock] = {}

    async def run_research(self, question: MetaculusQuestion) -> str:
        key = question.page_url
        # Fast path: already cached
        if key in CustomForecaster._research_cache:
            logger.info(f"Cache hit in CustomForecaster for URL {key}")
            return CustomForecaster._research_cache[key]
        # Ensure only one concurrent fetch per URL
        lock = CustomForecaster._research_locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Double-check cache inside lock
            if key in CustomForecaster._research_cache:
                logger.info(f"Cache hit in CustomForecaster for URL {key}; returning cached research.")
                return CustomForecaster._research_cache[key]
            # Perform actual fetch
            research = await super().run_research(question)
            CustomForecaster._research_cache[key] = research
            return research

    def __init__(self, *args, forecaster_description: str= "", forecaster_name: str= "", **kwargs):
        super().__init__(*args, **kwargs)
        self.forecaster_description = forecaster_description
        self.forecaster_name = forecaster_name
    
    def _build_base_prompt(self, question, research):
        """Creates the common base prompt sections used in all question types."""

        return f"""
            You are a professional forecaster interviewing for a job.

            Your forecasting reflects the forecasting style described between '<<<< >>>>':

            <<<<
            {self.forecaster_description}
            >>>>

            Make sure you take into account your forecasting style when producing your answer.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.
        """
    
    async def _process_forecast(
        self, 
        question: Any, 
        research: str, 
        prompt: str,
        extractor_fn: Callable[[str, Any], T]
    ) -> ReasonedPrediction[T]:
        """Common pipeline for processing forecasts of any type."""
        logger.info(f"Full prompt:\n{prompt}")
        
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction = extractor_fn(reasoning, question)
        

        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )
    
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        base_prompt = self._build_base_prompt(question, research)

        binary_specific = f"""
            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
        """
        
        prompt = clean_indents(base_prompt + binary_specific)
        
        return await self._process_forecast(
            question, 
            research, 
            prompt,
            lambda reasoning, _: PredictionExtractor.extract_last_percentage_value(
                reasoning, max_prediction=1, min_prediction=0
            )
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        base_prompt = self._build_base_prompt(question, research)
        
        multiple_choice_specific = f"""
            The options are: {question.options}
            
            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
        """
        
        prompt = clean_indents(base_prompt + multiple_choice_specific)
        
        return await self._process_forecast(
            question, 
            research, 
            prompt,
            lambda reasoning, q: PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, q.options
            )
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        base_prompt = self._build_base_prompt(question, research)
        
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        
        numeric_specific = f"""
            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}
            
            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
        """
        
        prompt = clean_indents(base_prompt + numeric_specific)
        
        return await self._process_forecast(
            question, 
            research, 
            prompt,
            lambda reasoning, q: PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, q
            )
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.forecaster_name!r}>"




if __name__ == "__main__":
    import argparse
    from typing import Literal
    import asyncio
    from forecasting_tools import MetaculusApi

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="test_questions",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = CustomForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        forecaster_description=forecaster_description,
        forecaster_name=model_name,
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": "openrouter/meta-llama/llama-4-maverick:free",
            "summarizer": "openrouter/meta-llama/llama-4-maverick:free",
        },
    )

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        # The new quarterly cup may not be initialized near the beginning of a quarter
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            # "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            # "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore
        
        
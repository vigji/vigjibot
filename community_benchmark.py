from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Literal
from pathlib import Path

import typeguard
from forecasting_tools import (
    Benchmarker,
    ForecastBot,
    GeneralLlm,
    MonetaryCostManager,
    MetaculusApi,
    ApiFilter,
    run_benchmark_streamlit_page,
)

from main import TemplateForecaster

logger = logging.getLogger(__name__)


async def benchmark_forecast_bot(mode: str) -> None:
    """
    Run a benchmark that compares your forecasts against the community prediction
    """

    number_of_questions = 1
    if mode == "display":
        run_benchmark_streamlit_page()
        return
    elif mode == "run":
        questions = MetaculusApi.get_benchmark_questions(number_of_questions)
    elif mode == "custom":
        # Below is an example of getting custom questions
        one_year_from_now = datetime.now() + timedelta(days=365)
        api_filter = ApiFilter(
            allowed_statuses=["open"],
            allowed_types=["binary"],
            num_forecasters_gte=40,
            scheduled_resolve_time_lt=one_year_from_now,
            includes_bots_in_aggregates=False,
            community_prediction_exists=True,
        )
        questions = await MetaculusApi.get_questions_matching_filter(
            api_filter,
            num_questions=number_of_questions,
            randomly_sample=True,
        )
        for question in questions:
            question.background_info = None  # Test ability to find new information
    else:
        raise ValueError(f"Invalid mode: {mode}")

    with MonetaryCostManager() as cost_manager:
        bots = [

            TemplateForecaster(
                predictions_per_research_report=1,
                use_mootlib=False,
                llms={
                    "default": "metaculus/anthropic/claude-3-7-sonnet-latest",
                    "summarizer": "openrouter/meta-llama/llama-4-maverick:free",
                },
            ),
            # TemplateForecaster(
            #     predictions_per_research_report=1,
            #     use_mootlib=True,
            #     mootlib_args={"n_results": 10, 
            #           "min_similarity": 0.5, 
            #           "exclude_platforms": ["Metaculus"]},
            #     llms={
            #         "default": "metaculus/anthropic/claude-3-7-sonnet-latest",
            #         "summarizer": "openrouter/meta-llama/llama-4-maverick:free",
            #     },
            # ),
            # TemplateForecaster(
            #     predictions_per_research_report=1,
            #     use_mootlib=False,
            #     llms={
            #         "default": "metaculus/openai/o3",
            #         "summarizer": "openrouter/meta-llama/llama-4-maverick:free",
            #     },
            # ),
            # TemplateForecaster(
            #     predictions_per_research_report=1,
            #     use_mootlib=True,
            #     mootlib_args={"n_results": 10, 
            #           "min_similarity": 0.5, 
            #           "exclude_platforms": ["Metaculus"]},
            #     llms={
            #         "default": "metaculus/openai/o3",
            #         "summarizer": "openrouter/meta-llama/llama-4-maverick:free",
            #     },
            # ),
            # TemplateForecaster(
            #     predictions_per_research_report=1,
            #     use_mootlib=False,
            #     llms={
            #         "default": "metaculus/openai/o4-mini",
            #         "summarizer": "openrouter/meta-llama/llama-4-maverick:free",
            #     },
            # ),
            # TemplateForecaster(
            #     predictions_per_research_report=1,
            #     use_mootlib=True,
            #     mootlib_args={"n_results": 10, 
            #           "min_similarity": 0.5, 
            #           "exclude_platforms": ["Metaculus"]},
            #     llms={
            #         "default": "metaculus/openai/o4-mini",
            #         "summarizer": "openrouter/meta-llama/llama-4-maverick:free",
            #     },
            # ),
            # TemplateForecaster(
            #     predictions_per_research_report=1,
            #     use_mootlib=False,
            #     llms={
            #         "default": "openrouter/meta-llama/llama-4-maverick:free",
            #         "summarizer": "openrouter/meta-llama/llama-4-maverick:free",
            #     },
            # ),
            # TemplateForecaster(
            #     predictions_per_research_report=1,
            #     use_mootlib=True,
            #     mootlib_args={"n_results": 10, 
            #           "min_similarity": 0.5, 
            #           "exclude_platforms": ["Metaculus"]},
            #     llms={
            #         "default": "openrouter/meta-llama/llama-4-maverick:free",
            #         "summarizer": "openrouter/meta-llama/llama-4-maverick:free",
            #     },
            # ),
        ]
        bots = typeguard.check_type(bots, list[ForecastBot])
        benchmarks = await Benchmarker(
            questions_to_use=questions,
            forecast_bots=bots,
            file_path_to_save_reports="benchmarks/",
            concurrent_question_batch_size=10,
        ).run_benchmark()
        for i, benchmark in enumerate(benchmarks):
            logger.info(f"Benchmark {i+1} of {len(benchmarks)}: {benchmark.name}")
            logger.info(f"- Final Score: {benchmark.average_expected_baseline_score}")
            logger.info(f"- Total Cost: {benchmark.total_cost}")
            logger.info(f"- Time taken: {benchmark.time_taken_in_minutes}")
        logger.info(f"Total Cost: {cost_manager.current_usage}")


if __name__ == "__main__":
    # Create log filename and ensure the directory exists
    log_filename = f"benchmarks/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    Path("benchmarks").mkdir(exist_ok=True)
    
    # Create and configure file handler directly
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Create and configure stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Test log messages
    logger.info("Logging system initialized")
    logger.info(f"Log file location: {log_filename}")

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark a list of bots")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "custom", "display"],
        default="display",
        help="Specify the run mode (default: display)",
    )
    args = parser.parse_args()
    mode: Literal["run", "custom", "display"] = args.mode
    asyncio.run(benchmark_forecast_bot(mode))

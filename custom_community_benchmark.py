from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Literal
from dotenv import load_dotenv

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

from custom_forecaster import forecaster_factory
from utils import load_forecasters_dict

logger = logging.getLogger(__name__)

forecasters_dict = load_forecasters_dict()

load_dotenv()

async def benchmark_forecast_bot(mode: str) -> None:
    """
    Run a benchmark that compares your forecasts against the community prediction.
    """

    number_of_questions = (
        50  # Recommend 100+ for meaningful error bars, but 30 is faster/cheaper
    )
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
            allowed_types=["binary"],  # "binary",
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
        mootlib_args={"n_results": 10, 
                      "min_similarity": 0.5, 
                      "exclude_platforms": ["Metaculus"]}
        
        bots = [
            forecaster_factory(model_name)(
                research_reports_per_question=1,
                predictions_per_research_report=1,
                use_research_summary_to_forecast=False,
                publish_reports_to_metaculus=False,
                folder_to_save_reports_to=None,
                skip_previously_forecasted_questions=False,
                use_mootlib=mootlib,
                mootlib_args=mootlib_args if mootlib else None,
                # forecaster_description=forecasters_dict[model_name],
                # forecaster_name=model_name,
                llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
                    # "summarizer": "openrouter/sophosympatheia/rogue-rose-103b-v0.2:free",# "openrouter/meta-llama/llama-4-maverick:free",
                    # "default": "openrouter/meta-llama/llama-4-maverick:free",
                    # "default": "openrouter/openai/gpt-4o-mini",
                    "default": model
                    # "default": "metaculus/openai/o4-mini",
                },
            )
            
            for model in ["metaculus/anthropic/claude-3-7-sonnet-latest",
                          # GeneralLlm(model="metaculus/openai/o4-mini", temperature=1, 
                          #                 timeout=40, allowed_tries=2),
                          "metaculus/openai/o3"
                          ]
            for model_name in list(forecasters_dict.keys())[:12] 
            for mootlib in [True]
            
        ]

        # bots = [forecaster_factory(list(forecasters_dict.keys())[0])(
        #         research_reports_per_question=1,
        #         predictions_per_research_report=1,
        #         use_research_summary_to_forecast=False,
        #         publish_reports_to_metaculus=False,
        #         folder_to_save_reports_to=None,
        #         skip_previously_forecasted_questions=False,
        #         llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
        #             # "default": "openrouter/" + model,
        #             "default": GeneralLlm(
        #                 model="openrouter/" + model,  #"metaculus/anthropic/claude-3-5-sonnet-20241022",  # metaculus/anthropic
        #                 temperature=0.3,
        #                 timeout=40,
        #                 allowed_tries=2,
        #             ),
        #         },

        #     ) for model in ["openrouter/meta-llama/llama-4-maverick:free",]
        #         #"openai/o4-mini",
        #                    # "openai/gpt-4.1-mini",
        #                    # "openai/gpt-4.1",
        #                     #"anthropic/claude-3.7-sonnet",
        #                     #"anthropic/claude-3.5-haiku:beta"]
        # ]
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                f"benchmarks/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
            ),
        ],
    )

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
        default="custom",
        help="Specify the run mode (default: display)",
    )
    args = parser.parse_args()
    mode: Literal["run", "custom", "display"] = args.mode
    asyncio.run(benchmark_forecast_bot(mode))

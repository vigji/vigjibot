import pandas as pd
from datetime import datetime
from tqdm import tqdm
from forecasting_tools.data_models.benchmark import BenchmarkForBot


def get_benchmark_df(benchmark):
    bot_config_data = benchmark.forecast_bot_config["llms"]["default"]
    # rint(bot_config_data)

    try:
        model_name = bot_config_data["original_model"]
    except:
        model_name = bot_config_data  # ["original_model"]
    model_name = model_name.split("openrouter/")[1]

    full_model_name = (
        benchmark.forecast_bot_class_name + "//" + model_name
    )  # (benchmark.forecast_bot_config["llms"]["default"]
    community_predictions = [
        rep.community_prediction for rep in benchmark.forecast_reports
    ]
    my_predictions = [rep.prediction for rep in benchmark.forecast_reports]
    num_forecasters = [
        rep.question.num_forecasters for rep in benchmark.forecast_reports
    ]
    question_ids = [rep.question.id_of_question for rep in benchmark.forecast_reports]
    question_texts = [rep.question.question_text for rep in benchmark.forecast_reports]

    model_df = pd.DataFrame(
        {
            "community_prediction": community_predictions,
            "my_prediction": my_predictions,
            "num_forecasters": num_forecasters,
            "question_id": question_ids,
            "question_text": question_texts,
            "bot_class": benchmark.forecast_bot_class_name,
            "bot_model": model_name,  # benchmark.forecast_bot_config["llms"]["default"]["original_model"].split("openrouter/")[1],
            "full_model_name": full_model_name,
        }
    )
    return model_df


def get_all_benchmark_df(benchmarks):
    all_benchmark_df = pd.concat(
        [get_benchmark_df(benchmark) for benchmark in benchmarks]
    )
    return all_benchmark_df


def _parse_benchmark_timestamp(filename):
    """Parse timestamp from benchmark filename."""
    timestamp_str = "_".join(filename.split("_")[1:]).split(".")[0]
    return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")


def get_all_runs_df(all_runs):
    all_runs_df = pd.DataFrame()
    for run in tqdm(all_runs):
        benchmarks = BenchmarkForBot.load_json_from_file_path(run)

        if len(benchmarks) > 1 and len(benchmarks[0].forecast_reports) > 28:
            timestamp = _parse_benchmark_timestamp(run)

            df = get_all_benchmark_df(benchmarks)
            df["run_id"] = run.split(".")[-2]
            df["timestamp"] = timestamp
            all_runs_df = pd.concat([all_runs_df, df])

    return all_runs_df

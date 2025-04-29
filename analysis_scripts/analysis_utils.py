import pandas as pd
from datetime import datetime
from tqdm import tqdm


# from forecasting_tools.forecast_helpers.benchmark_displayer import get_json_files
# from forecasting_tools.data_models.benchmark import BenchmarkForBot


def get_benchmark_df(benchmark):
    bot_config_data = benchmark.forecast_bot_config["llms"]["default"]
    # rint(bot_config_data)

    try:
        model_name = bot_config_data["original_model"]
    except:
        model_name = bot_config_data  # ["original_model"]

    if "openrouter/" in model_name:
        model_name = model_name.split("openrouter/")[1]
    else:
        model_name = model_name

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


def get_all_runs_df(all_runs, BenchmarkForBot):
    all_runs_df = pd.DataFrame()
    for i, run in tqdm(enumerate(all_runs)):
        print("loading run: ", run, " (", i, "/", len(all_runs), ")")
        benchmarks = BenchmarkForBot.load_json_from_file_path(run)

        # if len(benchmarks) > 1 and len(benchmarks[0].forecast_reports) > 28:
        timestamp = _parse_benchmark_timestamp(run)

        df = get_all_benchmark_df(benchmarks)
        df["run_id"] = run.split(".")[-2]
        df["timestamp"] = timestamp
        print(df.head())
        print(df["bot_class"].unique())

        all_runs_df = pd.concat([all_runs_df, df])
    
    print(len(all_runs_df["run_id"].unique()))

    return all_runs_df

if __name__ == "__main__":
    from forecasting_tools.forecast_helpers.benchmark_displayer import get_json_files
    from forecasting_tools.data_models.benchmark_for_bot import BenchmarkForBot
    from pathlib import Path
    data_path = Path(__file__).parent.parent / "benchmarks"
    assert data_path.exists()

    all_runs = get_json_files(data_path)
    all_runs_df = get_all_runs_df(all_runs, BenchmarkForBot)
    all_runs_df.to_csv("all_runs_df.csv")

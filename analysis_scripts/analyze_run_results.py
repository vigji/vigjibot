# %%
# Analyze run results

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

from pathlib import Path
from forecasting_tools.forecast_helpers.benchmark_displayer import get_json_files
from forecasting_tools.data_models.benchmark_for_bot import BenchmarkForBot

data_path = Path("./benchmarks")
assert data_path.exists()

all_runs = get_json_files(data_path)
print(len(all_runs))
print(all_runs[-1])


all_benchmarks = []
# for file in all_runs:
#    benchmarks = BenchmarkForBot.load_json_from_file_path(file)
#    all_benchmarks.append(benchmarks)

# print(len(all_benchmarks))

# for benchmark in all_benchmarks:
#     print(benchmark.explicit_name)
#     print(benchmark.forecast_reports[0].prediction)
# %%

for run in all_runs:
    benchmark = BenchmarkForBot.load_json_from_file_path(run)
    print(run, len(benchmark), len(benchmark[0].forecast_reports))
    # print(benchmark.explicit_name)
    # vprint(benchmark.forecast_reports[0].prediction)
# %%
sel_file = all_runs[-3]
benchmarks = BenchmarkForBot.load_json_from_file_path(sel_file)
# %%
benchmark = benchmarks[0]
print(run, len(benchmarks), len(benchmark.forecast_reports))
benchmark_id = sel_file.split(".")[-2]
benchmark_id
# %%
dict(benchmark)
# %%


# %%

all_df = get_all_runs_df(all_runs)
# %%
date_cutoff = datetime(2025, 4, 28)
model_name = "00VanillaForecaster"
df = all_df[(all_df.timestamp >= date_cutoff) & (all_df["bot_class"] == model_name)]


# %%
# use seaborn to scatter plot the community prediction vs my prediction for each "full_model_name", in separate subplots
fig, axs = plt.subplots(
    nrows=len(df["full_model_name"].unique()),
    ncols=1,
    figsize=(10, 10),
    sharex=True,
    sharey=True,
)
for i, full_model_name in enumerate(df["full_model_name"].unique()):
    sel_data = df[df["full_model_name"] == full_model_name]
    axs[i].scatter(sel_data["community_prediction"], sel_data["my_prediction"], s=5)
    # axs[i].set_title(full_model_name)
    p = 0.1
    lab = "\n".join(full_model_name.split("//"))
    axs[i].set(
        aspect="equal",
        ylabel=lab,
        xlabel="Community Prediction",
        ylim=(0 - p, 1 + p),
        xlim=(0 - p, 1 + p),
    )
    axs[i].xaxis.label.set_size(8)
    axs[i].yaxis.label.set_size(8)
plt.tight_layout()
plt.show()
# %%

len(df)

# %%
# check how many answers for each question id for each full_model_name
duplicated_answers = (
    df.groupby(["question_id", "full_model_name"]).size().reset_index(name="count")
)
duplicated_answers = duplicated_answers[duplicated_answers["count"] > 1]
len(duplicated_answers)
# %%
# Check max delta between repeated answers
max_delta = (
    df.groupby(["question_id", "full_model_name"])
    .apply(lambda x: x["my_prediction"].max() - x["my_prediction"].min())
    .reset_index(name="max_delta")
)
max_delta = max_delta[max_delta["max_delta"] > 0.1]
# %%
# Make a new df where duplicated answers are removed by picking the latest answer
no_dup_df = df.drop_duplicates(subset=["question_id", "full_model_name"], keep="last")
len(no_dup_df)

# Keep only question ids for which all full_model_names have one answer
no_dup_df = no_dup_df.groupby("question_id").filter(
    lambda x: len(x) == len(no_dup_df["full_model_name"].unique())
)
len(no_dup_df)

n_models = len(no_dup_df["full_model_name"].unique())
n_questions = len(no_dup_df["question_id"].unique())
# %%
# get average prediction pooling models
avg_pred = no_dup_df.groupby(["question_id"])["my_prediction"].mean().reset_index()
comm_avg_pred = (
    no_dup_df.groupby(["question_id"])["community_prediction"].mean().reset_index()
)
len(avg_pred), len(comm_avg_pred)
# %%
# Make a scatter plot of the average prediction vs the community prediction
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(comm_avg_pred["community_prediction"], avg_pred["my_prediction"], s=5)
ax.set(aspect="equal", xlabel="Community Prediction", ylabel="Average Prediction")
plt.show()
# %%
no_dup_df["question_text"].iloc[0]
# %%
benchmarks  # .forecast_reports#[0].question.question_text

# %%
benchmark.forecast_reports[0]
# %%

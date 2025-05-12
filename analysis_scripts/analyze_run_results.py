# %%
# Analyze run results

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from forecasting_tools.forecast_helpers.benchmark_displayer import get_json_files
from forecasting_tools.data_models.benchmark_for_bot import BenchmarkForBot
from traitlets import default
from analysis_utils import get_all_runs_df

data_path = Path(__file__).parent.parent / "benchmarks"
assert data_path.exists()

all_runs = get_json_files(data_path)
print(len(all_runs))
(all_runs)
# %%

for run in all_runs:
    pass
    # benchmark = BenchmarkForBot.load_json_from_file_path(run)
    # print(run, len(benchmark), len(benchmark[0].forecast_reports))
    # print(benchmark.explicit_name)
    # vprint(benchmark.forecast_reports[0].prediction)

if "all_runs_df.csv" not in os.listdir():
    all_df = get_all_runs_df(all_runs, BenchmarkForBot)
    all_df.to_csv("all_runs_df.csv")
else:
    all_df = pd.read_csv("all_runs_df.csv", index_col=0)
    all_df.timestamp = pd.to_datetime(all_df.timestamp)
print(all_df["bot_class"].unique())

df_questions = pd.read_csv("questions_df.csv", index_col=0)

all_df["question_cluster"] = (
    all_df["question_id"]
    .map(df_questions.set_index("question_id")["cluster"])
    .fillna(value=-1)
    .astype(int)
    .astype(str)
)
# %%
date_cutoff = datetime(2025, 4, 28)
model_name = "00VanillaForecaster"
df_models = all_df[
    (all_df.timestamp >= date_cutoff) & (all_df["bot_class"] == model_name)
]
df_models

# Define colors for question clusters

# %%
# %%
# use plotly to create an interactive scatter plot with all models

# Create a figure with all models
fig = go.Figure()
colors = px.colors.qualitative.Set3[:10] + [
    "#000000"
]  # Using Set3 palette from plotly plus black

# Add traces for each model with low alpha
for model_name in df_models["bot_model"].unique():
    sel_data = df_models[df_models["bot_model"] == model_name]

    # Create hover text with all model predictions for each question
    hover_texts = []
    for _, row in sel_data.iterrows():
        question_id = row["question_id"]
        other_models = df_models[df_models["question_id"] == question_id]
        # Sort models by name and create predictions text
        sorted_models = sorted(
            zip(other_models["bot_model"], other_models["my_prediction"]),
            key=lambda x: x[0],
        )
        model_predictions = "<br>".join([f"{m}: {p:.2f}" for m, p in sorted_models])
        hover_texts.append(
            f"Question: {row['question_text']}<br>"
            f"Community: {row['community_prediction']:.2f}<br>"
            f"Cluster: {row['question_cluster']}<br>"
            f"All model predictions:<br>{model_predictions}"
        )

    cols = [colors[int(cluster)] for cluster in sel_data["question_cluster"]]
    fig.add_trace(
        go.Scatter(
            x=sel_data["community_prediction"],
            y=sel_data["my_prediction"],
            mode="markers",
            name=model_name,
            text=hover_texts,
            hoverinfo="text",
            marker=dict(
                size=8,
                opacity=0.3,
                color=cols,
                line=dict(width=0, color="black"),  # Default no border
            ),
            visible=True,
        )
    )

# Create buttons for the dropdown
buttons = []
for i, model_name in enumerate(df_models["bot_model"].unique()):
    # Create a list of marker opacity values for all traces
    opacities = [0.3] * len(df_models["bot_model"].unique())
    opacities[i] = 1.0  # Set selected model to full opacity

    # Create lists for line widths (0 for unselected, 1 for selected)
    line_widths = [0] * len(df_models["bot_model"].unique())
    line_widths[i] = 1

    buttons.append(
        dict(
            label=model_name,
            method="update",
            args=[
                {
                    "visible": [True] * len(df_models["bot_model"].unique()),
                    "marker.opacity": opacities,
                    "marker.size": [
                        8 if j != i else 10
                        for j in range(len(df_models["bot_model"].unique()))
                    ],
                    "marker.line.width": line_widths,  # Update line widths
                }
            ],
        )
    )

# Add dropdown
fig.update_layout(
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.1,
            y=1.1,
        )
    ],
    title="Model Predictions vs Community Predictions",
    xaxis_title="Community Prediction",
    yaxis_title="Model Prediction",
    showlegend=True,
    height=800,
    width=1200,  # Wider figure
    xaxis=dict(
        scaleanchor="y",  # This ensures the x-axis is scaled to match the y-axis
        scaleratio=1,  # This ensures a 1:1 aspect ratio
        range=[-0.1, 1.1],  # Fixed range to ensure consistent view
        constrain="domain",  # This ensures the aspect ratio is maintained
    ),
    yaxis=dict(
        range=[-0.1, 1.1],  # Fixed range to ensure consistent view
        constrain="domain",  # This ensures the aspect ratio is maintained
    ),
    margin=dict(l=50, r=50, t=100, b=50),  # Add some margin for better visibility
)

# Add diagonal line
fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))

fig.show()
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
# %%
date_cutoff = datetime(2025, 4, 26)
model_name = "00VanillaForecaster"
# df_models = all_df[(all_df.timestamp >= date_cutoff)]

# %%
# for each value of run_id, show number of unique combinations of bot_class and question_id
all_df.groupby("run_id").apply(
    lambda x: (len(x["bot_class"].unique()), len(x["question_id"].unique()))
)  # .value_counts()
# %%
run_id = "/Users/vigji/code/vigjibot/benchmarks/benchmarks_2025-04-28_12-42-06"
df_models = all_df[all_df["run_id"] == run_id]
df_models
# %%
fig = go.Figure()

colors = px.colors.qualitative.Set3[:10] + [
    "#000000"
]  # Using Set3 palette from plotly plus black

dropdown_on = "bot_class"

# Let's start from a simple plot. I select a model using the dropdown, and the plots diplaied
# are updated by filtering only on that model.


# Simple plot with dropdown to select models
fig_simple = go.Figure()
dropdown_on = "bot_class"

# Get unique models
models = df_models[dropdown_on].unique()

# Add traces for each model (initially hidden)
for model in models:
    model_data = df_models[df_models[dropdown_on] == model]
    fig_simple.add_trace(
        go.Scatter(
            x=model_data["community_prediction"],
            y=model_data["my_prediction"],
            mode="markers",
            name=model,
            visible=False,  # All traces start hidden
        )
    )

# Create dropdown buttons
buttons = []
for i, model in enumerate(models):
    visibility = [False] * len(models)
    visibility[i] = True
    buttons.append(dict(label=model, method="update", args=[{"visible": visibility}]))

# Update layout with dropdown
fig_simple.update_layout(
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.1,
            y=1.1,
        )
    ],
    title="Simple Model Predictions vs Community Predictions",
    xaxis_title="Community Prediction",
    yaxis_title="Model Prediction",
    showlegend=True,
)

# Show the first model by default
fig_simple.data[0].visible = True

fig_simple.show()

# %%

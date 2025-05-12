# %%

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

df_file = Path("/Users/vigji/code/vigjibot/data/combined_markets.csv")
df = pd.read_csv(df_file)

# %%
len(df)

# %%
df.source_platform.value_counts()
# %%

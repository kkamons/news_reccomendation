import pandas as pd
import numpy as np
import sys
from pathlib import Path, PureWindowsPath


def gen_reader_sample(n, com):

	sample = np.random.choice(a=[-1, -0.5, 0, 0.5, 1],
								size = n,
								p=[0.09, 0.18, 0.38, 0.26, 0.09])
	reader_df = pd.DataFrame(data=sample,columns = ["leaning"])
	reader_df["id"] = reader_df.index
	sources = com["source"].unique()

	lean_map = {-1:"left", -0.5:"lean left", 0:"center", 0.5:"lean right", 1:"right"}

	reader_df = pd.concat([reader_df, pd.DataFrame(columns=sources)], sort=True)

	for src in sources:
		reader_df.loc[reader_df.leaning == -1, src] = com.loc[com.source == src, lean_map[-1]].values[0]
		reader_df.loc[reader_df.leaning == -0.5, src] = com.loc[com.source == src, lean_map[-0.5]].values[0]
		reader_df.loc[reader_df.leaning == 0, src] = com.loc[com.source == src, lean_map[0]].values[0]
		reader_df.loc[reader_df.leaning == 0.5, src] = com.loc[com.source == src, lean_map[0.5]].values[0]
		reader_df.loc[reader_df.leaning == 1, src] = com.loc[com.source == src, lean_map[1]].values[0]

		# reader_df.loc[reader_df.leaning == -1, src]
		# says: "select src from reader_df where leaning = -1"

	return reader_df

def gen_src_bias(src):
	src.columns = ["source", "type", "bias", "url_src", "url_all"]
	src = src.loc[src["bias"]!="Mixed"]

	bias_map = {"Left": -1, "Lean Left": -0.5, "Center":0, "Lean Right": 0.5, "Right": 1}
	src = src.replace({"bias":bias_map})

	return src[["source", "bias"]]

def main():

	demo_path = Path("projdata\\Voter_Distribution_2016\\Voter_demo_analysis_CSV.csv")
	all_path = Path("projdata\\Sources_Political_Leanings\\all.csv")
	trust_path= Path("projdata\\Trust_In_Media.csv")
	n=1000

	demos = pd.read_csv(demo_path, encoding='unicode_escape')
	sources = pd.read_csv(all_path, encoding='unicode_escape')
	media_trust = pd.read_csv(trust_path, encoding='unicode_escape')
	media_trust.columns = ["source", "left", "lean left", "center", "lean right", "right"]
	src_bias_df = gen_src_bias(sources)
	common = media_trust.merge(src_bias_df, on="source")
	print(common)
	reader_df = gen_reader_sample(n, common)



if __name__ == "__main__":
	main()
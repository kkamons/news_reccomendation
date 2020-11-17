import pandas as pd
import numpy as np
import sys
from pathlib import Path, PureWindowsPath

lean_map = {-1:"left", -0.5:"lean left", 0:"center", 0.5:"lean right", 1:"right"}
def gen_reader_sample(n, com):

	sample = np.random.choice(a=[-1, -0.5, 0, 0.5, 1],
								size = n,
								p=[0.09, 0.18, 0.38, 0.26, 0.09])
	reader_df = pd.DataFrame(data=sample,columns = ["leaning"])
	reader_df["reader_id"] = reader_df.index
	sources = com["source"].unique()


	reader_df = pd.concat([reader_df, pd.DataFrame(columns=sources)], sort=True)
	leanings = [-1, -0.5, 0, 0.5, 1]
	for src in sources:
		for l in leanings:
			reader_df.loc[reader_df.leaning == l, src] = com.loc[com.source == src, lean_map[l]].values[0]

		# reader_df.loc[reader_df.leaning == -1, src]
		# says: "select src from reader_df where leaning = -1"

	return reader_df
#Currently this function is dumb and just randomly selects a media outlet to recommend to a reader
# Later this could be changed to recommend specific sources based of the readers political leaning
def getSource(media_df):
	sources = media_df["source"]
	return np.random.choice(a=sources)


# Returns the probability of a viewer reading an article base off the media outlet
def calc_prob_reading(reader_leaning, reader_trust, source_leaning):
	variance = 0.05
	#If the viewer trusts the media outlet, they are more likely to read it
	#If the farther the media outlet sits from the viewer_leaning, the less likely they will be to read
	# prob = np.random.normal(reader_trust- abs(reader_leaning-source_leaning), variance,1)[0]
	noise = np.random.normal(0,variance,1)[0]
	prob = (reader_trust - abs(reader_leaning-source_leaning) + noise)/(reader_trust + abs(reader_leaning-source_leaning) + abs(noise))
	if prob < 0:
		prob = 0
	return prob

def gen_src_bias(src):
	src.columns = ["source", "type", "bias", "url_src", "url_all"]
	src = src.loc[src["bias"]!="Mixed"]

	bias_map = {"Left": -1, "Lean Left": -0.5, "Center":0, "Lean Right": 0.5, "Right": 1}
	src = src.replace({"bias":bias_map})

	return src[["source", "bias"]]

def main():
	mac=True
# Big sad the path shit dont work but its fine
	if(mac):
		demo_path = Path("projdata//Voter_Distribution_2016//Voter_demo_analysis_CSV.csv")
		all_path = Path("projdata//Sources_Political_Leanings//all.csv")
		trust_path= Path("projdata//Trust_In_Media.csv")
	else:
		print("Windows is dumb")
		demo_path = Path("projdata\\Voter_Distribution_2016\\Voter_demo_analysis_CSV.csv")
		all_path = Path("projdata\\Sources_Political_Leanings\\all.csv")
		trust_path= Path("projdata\\Trust_In_Media.csv")
	n=1000

	demos = pd.read_csv(demo_path, encoding='unicode_escape')
	sources = pd.read_csv(all_path, encoding='unicode_escape')
	media_trust = pd.read_csv(trust_path, encoding='unicode_escape')
	media_trust.columns = ["source", "left", "lean left", "center", "lean right", "right"]
	src_bias_df = gen_src_bias(sources)
	#This holds a frame with all the media outlets we will be using
	media_frame = media_trust.merge(src_bias_df, on="source")
	print(media_frame)
	#This will generate a sample of readers
	reader_df = gen_reader_sample(n, media_frame)
	print(reader_df.head())
	# reader_df.loc[reader_df.leaning == -1, src]
	# says: "select src from reader_df where leaning = -1"
	readers = reader_df["reader_id"].unique()
	# for reader in readers:
		# Get the source to recommend to the reader
	reader = 0.0
	rec_source = getSource(media_frame)
	print("Source: ",rec_source)
	# Calc prob of reader reading source
	reader_leaning = reader_df.leaning[reader]
	print("Reader Leaning:",reader_leaning, lean_map[reader_leaning])
	reader_trust = media_frame.loc[media_frame.source == rec_source, lean_map[reader_leaning]].values[0]
	print("Reader Trust in Source: ",reader_trust)
	source_leaning = media_frame.loc[media_frame.source == rec_source, "bias"].values[0]
	print("Source Bias: ",source_leaning)

	prob_to_read = calc_prob_reading(reader_leaning, reader_trust, source_leaning)
	print("Probability to Read: ", prob_to_read)
	source_is_read = np.random.choice(a=[True,False],size=1,p=[prob_to_read,1-prob_to_read])[0]
	print(source_is_read)
# TODO:
# X CalcProbOfReading()
# CalcProbOfTrusting()
# ReCalcTrustLevel() - calculate the trust level of the recommended media outlet
# ReCalcPolLeaning() - base off leaning of media outlet and whether they trusted, if they trusted then it will effect their belief in the direction of the media outlet, if they dont it may polarize them.


if __name__ == "__main__":
	main()
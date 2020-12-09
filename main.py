import pandas as pd
import numpy as np
import sys
import math
from matplotlib import pyplot as plt
from pathlib import Path, PureWindowsPath

show_fig = True
num_trials = 500
num_readers=1000
debug = False
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

def recMoreModerate(media_df, reader_leaning):
	rounded_leaning = round(reader_leaning*2)/2 # if we want to use the map, then we need to round the leaning
	src_bias = [-0.5, 0, 0.5]
	# print(rounded_leaning)
	# if we want readers to tend towards center then we should select sources that are more center than where they are:
	if rounded_leaning != 0:
		# because we can go out of range <-1 and >1
		if rounded_leaning < 0:
			src_bias = [-0.5,0]
		else:
			src_bias = [0,0.5]
	# print(src_bias)
	sources = media_df["source"].loc[media_df["bias"].isin(src_bias)]
	# sources = media_df["source"]
	return np.random.choice(a=sources)

def recMostLikelyToRead(media_df, reader_leaning):
	rounded_leaning = round(reader_leaning*2)/2 # if we want to use the map, then we need to round the leaning
	sources = media_df["source"].loc[media_df["bias"].isin(rounded_leaning)]
	return np.random.choice(a=sources)

#Currently this function is dumb and just randomly selects a media outlet to recommend to a reader
# Later this could be changed to recommend specific sources based of the readers political leaning
def getSource(media_df, reader_leaning,recMethod):
	if recMethod == "Moderate":
		rec = recMoreModerate(media_df, reader_leaning)
	elif recMethod == "Random":
		rec = np.random.choice(media_df["source"])
	elif recMethod == "LikelyToRead":
		rec = np.random.choice(media_df["source"])
	else:
		rec = np.random.choice(media_df["source"])
	return rec

def gen_src_bias(src):
	src.columns = ["source", "type", "bias", "url_src", "url_all"]
	src = src.loc[src["bias"]!="Mixed"]

	bias_map = {"Left": -1, "Lean Left": -0.5, "Center":0, "Lean Right": 0.5, "Right": 1}
	src = src.replace({"bias":bias_map})

	return src[["source", "bias"]]

# Returns the probability of a viewer reading an article base off the media outlet
def calc_prob_reading(reader_leaning, reader_trust, source_leaning):
	variance = 0.05
	#If the viewer trusts the media outlet, they are more likely to read it
	#If the farther the media outlet sits from the viewer_leaning, the less likely they will be to read
	# prob = np.random.normal(reader_trust- abs(reader_leaning-source_leaning), variance,1)[0]
	noise = np.random.normal(0,variance,1)[0]

	# prob = 1-((abs(reader_leaning - source_leaning)/2)+0.5)+ noise
	# sqrt((a-b)^2)

	if abs(reader_leaning-source_leaning) < 0.5:
		prob = 1-((abs(reader_leaning - source_leaning)/2))+ noise + reader_trust/2
	else:
		prob = noise
	if prob > 0.5:
		prob = min(prob,1)
	else:
		prob = max(prob,0)

	prob = np.random.binomial(1,prob,1)[0]
	# prob = (reader_trust - abs(reader_leaning-source_leaning) + noise)#/(reader_trust + abs(reader_leaning-source_leaning) + abs(noise))
	'''
	p(read | trust) = p(trust | read)p(read)/p(trust)
	p(trust | read)= reader_trust, p(read) = 1-abs(reader_leaning-source_leaning)+noise, p(trust) = 
	'''
	# print(prob)
	# if prob < 0:
	# 	prob = 0
	# if prob > 1:
	# 	prob = 1	
	return prob


def calc_prob_trusting(reader_leaning, reader_trust, source_leaning):
	variance = 0.05
	noise = np.random.normal(0,variance,1)[0]
	# prob = 1-((abs(reader_leaning - source_leaning)/2)+0.5)+ noise
	prob = (reader_trust - abs(reader_leaning-source_leaning) + noise)#/(reader_trust + abs(reader_leaning)+abs(source_leaning) + abs(noise))
	if prob < 0:
		prob = 0
	if prob > 1:
		prob = 1
	return prob

def recalc_trust(reader_trusts_story, reader_trust):
	if reader_trusts_story:
		new_reader_trust = reader_trust + 0.01
	else:
		new_reader_trust = reader_trust - 0.01
	# print(new_reader_trust)
	return new_reader_trust

def recalc_leaning(reader_leaning, source_leaning, reader_trust):   #(reader_leaning, new_reader_trust,source_leaning):
	# if (reader_leaning - source_leaning) <=0.5:
	# 	return reader_leaning + np.sign(reader_leaning)*0.05
	# elif (reader_leaning - source_leaning) < 0.1:
	# 	return reader_leaning + np.sign(reader_leaning)*0.1
	# else:
	# 	return reader_leaning
	variance = 0.05
	noise = np.random.normal(0,variance,1)[0]
	new_leaning = reader_leaning + (source_leaning - reader_leaning)/2*.1 + noise
	if new_leaning > 1:
		new_leaning = 1
	elif new_leaning < -1:
		new_leaning = -1
	return new_leaning
	# return reader_leaning + ((np.sign(source_leaning)))*0.1 + noise

	
	# return (reader_leaning - source_leaning)
	# return source_leaning*new_reader_trust


def gen_before_after(first, last):

	bins1 = np.linspace(math.ceil(min(first)),
		math.floor(max(first)),
		10)
	bins2 = np.linspace(min(last),
		max(last),
		10)

	fig = plt.figure()
	plt.subplot(1,2,1)
	plt.xlim([min(first), max(first)])
	plt.hist(first, bins = bins1, alpha = 0.5)
	plt.title("Before")
	plt.subplot(1,2,2)
	plt.xlim([min(last), max(last)])
	plt.hist(last, bins = bins2, alpha = 0.5)
	plt.title("After")
	fig.suptitle("{} trails for {} readers".format(num_trials, num_readers))
	plt.show()


def main():
	mac=True
# Big sad the path shit dont work but its fine
	if(mac):
		demo_path = Path("projdata//Voter_Distribution_2016//Voter_demo_analysis_CSV.csv")
		all_path = Path("projdata//Sources_Political_Leanings//all.csv")
		trust_path= Path("projdata//Trust_In_Media.csv")
	else:
		demo_path = Path("projdata\\Voter_Distribution_2016\\Voter_demo_analysis_CSV.csv")
		all_path = Path("projdata\\Sources_Political_Leanings\\all.csv")
		trust_path= Path("projdata\\Trust_In_Media.csv")
	
	

	demos = pd.read_csv(demo_path, encoding='unicode_escape')
	sources = pd.read_csv(all_path, encoding='unicode_escape')
	media_trust = pd.read_csv(trust_path, encoding='unicode_escape')
	media_trust.columns = ["source", "left", "lean left", "center", "lean right", "right"]
	src_bias_df = gen_src_bias(sources)
	#This holds a frame with all the media outlets we will be using
	media_frame = media_trust.merge(src_bias_df, on="source")
	# print(media_frame)
	#This will generate a sample of readers
	reader_df = gen_reader_sample(num_readers, media_frame)
	# print(reader_df.head())
	# reader_df.loc[reader_df.leaning == -1, src]
	# says: "select src from reader_df where leaning = -1"
	readers = reader_df["reader_id"].unique()
	# print('+++++++++++++++++ ', reader_df.head())
	# print(reader_df)
	leaning_prog = []
	for i in range(num_trials):
		if debug: print('________________________________________New Trial__________________________________________________')
		if debug: print(reader_df.leaning.values)
		leaning_prog.append(list(reader_df.leaning.values))
		for reader in readers:
			if debug: print("============================ new reader ======================================")
		# Get the source to recommend to the reader
		# reader = 0.0
			reader_leaning = reader_df.leaning[reader]
			rounded_leaning = round(reader_leaning*2)/2 # if we want to use the map, then we need to round the leaning
			rec_source = getSource(media_frame, reader_leaning, "Moderate")
			if debug: print("Source: ",rec_source)
			# Calc prob of reader reading source
			if debug: print("Reader Leaning:",reader_leaning)#, lean_map[reader_leaning])
			# reader_trust = media_frame.loc[media_frame.source == rec_source, lean_map[reader_leaning]].values[0]
			# reader_trust = reader_df.loc[reader_df.reader_id == reader, rec_source].values[0]
			reader_trust = media_frame.loc[media_frame.source == rec_source, lean_map[rounded_leaning]].values[0]


			if debug: print("Reader Trust in Source: ",reader_trust)
			source_leaning = media_frame.loc[media_frame.source == rec_source, "bias"].values[0]
			if debug: print("Source Bias: ",source_leaning)

			source_is_read = calc_prob_reading(reader_leaning, reader_trust, source_leaning)
			# if debug: print("Probability to Read: ", prob_to_read)
			# source_is_read = np.random.choice(a=[True,False],size=1,p=[prob_to_read,1-prob_to_read])[0]
			if(source_is_read):
				if debug: print("Article Read")
				#calc prob of trusting
				# prob_trust = calc_prob_trusting(reader_leaning, reader_trust, source_leaning)
				# if debug: print("Probability to Trust Source: ", prob_trust)
				# reader_trusts_story = np.random.choice(a=[True,False],size=1,p=[prob_trust,1-prob_trust])[0]
				#recalc trust level
				# new_trust_level = recalc_trust(reader_trusts_story, reader_trust)
				# reader_df.loc[reader_df.reader_id == reader, rec_source] = new_trust_level
				#reCalcPolLeaning - im thinking this should be harder to change than the trust level in the media outlet
				reader_df.loc[reader_df.reader_id == reader, "leaning"] = recalc_leaning(reader_leaning, source_leaning, reader_trust)#new_trust_level,source_leaning)
				if debug: print("New Leaning: ", reader_df.loc[reader_df.reader_id == reader, "leaning"].values[0])
				
			else:
				if debug: print("Fake news!")

		min_lean = reader_df.leaning.min()
		max_lean = reader_df.leaning.max()

		# keep it between -1 and 1
		reader_df["leaning"] = 2*(reader_df["leaning"])/(max_lean-min_lean)
	# print(reader_df.bias.values)
	# print("the thing")
	# percent_left = len(reader_df[reader_df.bias < -0.75].bias.values)/n
	# condition = (reader_df.bias < -0.25 and reader_df.bias >= -0.75)
	# print("condition: ",condition)
	# percent_lean_left = len(reader_df[condition].bias.values)/n
	# percent_center = len(reader_df[reader_df.bias < 0.25 and reader_df.bias >= -0.25].bias.values)/n
	# percent_lean_right = len(reader_df[reader_df.bias < 0.75 and reader_df.bias >= 0.25].bias.values)/n
	# percent_right = len(reader_df[reader_df.bias >= 0.75].bias.values)/n
	# print(percent_left)
	# print(percent_lean_left)
	# print(percent_center)
	# print(percent_lean_right)
	# print(percent_right)
	# print(reader_df[reader_df.bias > 0.5].bias.values)

	if show_fig: gen_before_after(leaning_prog[0], leaning_prog[-1])

	
# TODO:
# X CalcProbOfReading()
# X CalcProbOfTrusting()
# X ReCalcTrustLevel() - calculate the trust level of the recommended media outlet
# X ReCalcPolLeaning() - base off leaning of media outlet and whether they trusted, if they trusted then it will effect their belief in the direction of the media outlet, if they dont it may polarize them.
# Make the algorithm less dumb
# recalculate the percentage in each quandrant
# Print a histogram of distribution before and after the stuff
# NORMALIZE Probabilites


if __name__ == "__main__":
	main()
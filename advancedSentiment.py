
import nltk
import nltk.sentiment.util 
import nltk.sentiment.sentiment_analyzer
from  nltk.sentiment.vader import SentimentIntensityAnalyzer




def MySentimentAnalyzer():

	def score_feedback(text):
		positive_words = ['love','genuine','liked']
		if "_NEG" in ''.join(nltk.sentiment.util.mark_negation(text.split())):
			score = -1 
		else:
			analyzer = nltk.sentiment.util.extract_unigram_feats(text.split(),positive_words)
			if True in analyzer.values():
				score = 1 
			else:
				score = 0
		return score 

	feedback = """Add your own custom text here."""

	print("------ Custom Scorer -------")
	for text in feedback.split("\n"):
    	    print("\n Score = {} for >> {}".format(score_feedback(text),text))

def AdvancedSentimentAnalyzer():

	sentences = [
'This is a positive sample text',
'You can add your own sample text here'
	]
	#sentences = word.getTextWord(FileName).split('.')
	senti = SentimentIntensityAnalyzer()
	print("------- Built-in Sentiment Analyzer ---------")
	for sentence in sentences:
		print('[{}]'.format(sentence),'\n --->')
		kvp = senti.polarity_scores(sentence)
		for k in kvp:
			print("{} -- {}".format(k, kvp[k]),'\n')
		print()

if __name__ == "__main__":
	AdvancedSentimentAnalyzer()
	MySentimentAnalyzer()


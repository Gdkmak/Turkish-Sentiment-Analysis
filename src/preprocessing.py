from . import correction
from . import utils
import csv

STOPWORDS = []


with open('./src/stopwords-tr.csv', 'r', encoding='utf-8') as ff:
  read = csv.reader(ff, delimiter = '\n')
  for i in read: 
    STOPWORDS.extend(i)


def preprocess(text):
	delete_list = [",", "’"]
	tweet = utils.delete_characters_space(text, delete_list)
	word_list = tweet.split() 
	word_list = [utils.stem_word(correction.correction( \
                              utils.remove_punct(utils.remove_repeating_char(utils.remove_with_regex(word))))) \
                              for word in word_list]
	word_list = [word for word in word_list if len(word) > 1]
	word_list = utils.remove_words(word_list, STOPWORDS)

	sentence = ""
	for word in word_list:
		sentence = sentence + " " + word

	return(sentence)
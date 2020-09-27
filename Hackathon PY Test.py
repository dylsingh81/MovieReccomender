import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
amountOfMovies = -10000

#Adapated from https://towardsdatascience.com/how-to-build-from-scratch-a-content-based-movie-recommender-with-natural-language-processing-25ad400eb243
#
df = pd.read_csv('data.csv')

df = df[['Title','Origin/Ethnicity','Genre','Director','Cast','Plot']]
df.head()

df = df.dropna()
df = df.loc[df['Genre']!="unknown"]
df = df.loc[df['Director']!="Unknown"]
df = df.loc[df['Origin/Ethnicity']=="American"]
df = df.iloc[amountOfMovies:]
df = df.set_index("Title")

# initializing the new column
df['Key_words'] = ""

for index, row in df.iterrows():
    plot = row['Plot']
    
    # instantiating Rake, by default it uses english stopwords from NLTK
    # and discards all puntuation characters as well
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(plot)

    # getting the dictionary whith key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()
    
    # assigning the key words to the new column for the corresponding movie
    row['Key_words'] = list(key_words_dict_scores.keys())

# dropping the Plot column
df.drop(columns = ['Plot'], inplace = True)

#for words in in df['Key_words']:
#  print(wor)
#keywords = ' '.join(word[0] for word in df['Key_words'])
#print(keywords)

for index, row in df.iterrows():
  row['Key_words'] = ' '.join(word for word in row['Key_words'])

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['Key_words'])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use in the function to match the indexes
indices = pd.Series(df.index)
#  defining the function that takes in movie title 
# as input and returns the top 10 recommended movies
def recommendations(title, cosine_sim = cosine_sim):
    
    # initializing the empty list of recommended movies
    recommended_movies = []
    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
        
    return recommended_movies

userInput = ""
while(userInput !="Q"):
    print("\nInput a movie title to get reccomendations (Press Q to Quit): ")
    userInput = input()
    if(userInput.upper() == "Q"):
        exit(0)

    ratios = []
    for i in indices:
        ratios.append((fuzz.ratio(userInput, i), i))

    ratios.sort(reverse=True)
    if(ratios[0][0] != 100):
        print("Did you mean (Y/N): ", ratios[0][1])
        yn = input()
        if yn.upper() == "N" or yn.upper() == "NO":
            print("Please enter a different title")
            continue
        elif yn.upper() == "Y" or yn.upper() == "YES":
            userInput = ratios[0][1]
        else:
            print("Y/N not entered.")
            continue
    try:
        print(recommendations(userInput))
    except:
        print("Title error")

"""
The flask application package.
"""

from flask import Flask,request, render_template
app = Flask(__name__)

"""
Movie recomendation code 

"""


import FlaskWebProject1.views
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from flask import jsonify
import os 
import pickle 

#for matching the nearest movie name to the  user input name
from fuzzywuzzy  import process

#credits = pd.read_csv(r"C:\Users\Yerra\Desktop\spark\movie_recommendation_system\data\tmdb_5000_credits.csv")
#movies = pd.read_csv(r".\data\movie_dataset.csv")
movies = pd.read_csv(os.path.join(os.path.abspath(r"."),"data", "movie_dataset.csv"))
movie_list = list(movies['original_title'].unique())
features = ['keywords','cast','genres','director']

# We are considering the 4 features i.e. keywords, cast, genres and director for the movie reocomedation 
def combine_features(row):
    return row['keywords']+' '+row['cast']+' '+row['genres']+' '+row['director']

# Helper function to get the title from the index
def get_title_from_index(index):
    return movies[movies.index == index]['title'].values[0]

# Helper function to get the index from the Title
def  get_index_from_title(title):
    return movies[movies.title == title]['index'].values[0]

def movie_predict(movie_user_like):
    for feature in features:
        movies[feature] = movies[feature].fillna('')

    movies['combined_features'] = movies.apply(combine_features, axis = 1)

    # Convert a collection of text into a  matrix of token counts
    count_matrix = CountVectorizer().fit_transform(movies['combined_features'])

    # Get the cosine similarity matrix from the count matrix
    cosine_sim = cosine_similarity(count_matrix)

    #Get the Movie name from the user and fetch the index from the movie
    movie_name = process.extractOne(movie_user_like, movies['title'])
    movie_index = get_index_from_title(movie_name[0])

    ## Enumerate all the simillarty score for the movie to make a tuple of movie index and similarity scores 
    # Note : we will return a list of tuples in the form 

    similar_movies = list(enumerate(cosine_sim[movie_index]))

    # Now sort the similar movies based on the similarity score  in descinding order and fetch only the top 5 matching movies and remove the 1st record since it matches with it's own record

    recomended_movies =  []

    for i in sorted(similar_movies, key = lambda x: x[1], reverse= True)[1:6]:
        recomended_movies.append(get_title_from_index(i[0]))


    return recomended_movies


# End of the Movie recomendation system

###################################################
# Start of the of the Movie Recomendation system #
###################################################
import wikipedia

def GetWikipediaData(title_name):
    try:
        description = wikipedia.page(title_name).content
        description = description.split('\n\n')[0]
        image = 'https://www.wikipedia.org/static/apple-touch/wikipedia.png'
        images = wikipedia.page(title_name).images
        for image in images:
            if('poster' in image.lower()):
                break;            
    except:
        description = " No wikipedia Description avaialbe"
        image = 'https://www.wikipedia.org/static/apple-touch/wikipedia.png'
        pass
    return(description, image)


# End of the Recomendation system

@app.route("/movie", methods=["GET", "POST"])
def movie():
    feedback = 'outside of the Post method'
    if request.method == "POST":

        req = request.form['movie']        
        movie_predicted =  movie_predict(req)    

        # Added to fetch the Wiki data
        wiki_movie = {}
        for i,j in enumerate(movie_predicted):
            wiki_movie_description, wiki_movie_poster  = GetWikipediaData(j +' film') 
            wiki_movie[i] = [j, wiki_movie_description, wiki_movie_poster]
              
        return render_template("recommended.html",  movie_predicted = movie_predicted,movie_list = movie_list, req= req, wiki_movie = wiki_movie, title='Movie Recommendation')

    return render_template("recommended.html",movie_list = movie_list, title='Movie Recommendation')



@app.route("/test", methods=["GET", "POST"])
def test():
    # Load the Default Wines paramaters values  from the Json file min, max, mean        
    return render_template("test.html", title='Wine Quality Predictor')
    

    
@app.route("/backgroud_process", methods = ["GET","POST"])
def backgroud_process():
    defalut_wines_values =  json.load(open(os.path.join(os.path.abspath(r"."),"FlaskWebProject1","static","data","default_val_fixed.json"))) 

    #Load the Saved model  Gradient Boosting model for wine Recomendation system
    gb_model = pickle.load(open(os.path.join(os.path.abspath(r"."),"models","gb_model_dump.p"), "rb"))

    if request.method == "POST":

        type                =  float(request.form['type'])
        fixed_acidity       =  float(request.form['fixed_acidity'])
        volatile_acidity    =  float(request.form['volatile_acidity'])
        citric_acid         =  float(request.form['citric_acid'])
        residual_sugar      =  float(request.form['residual_sugar'])
        chlorides           =  float(request.form['chlorides'])
        free_sulfur_dioxide =  float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide=  float(request.form['total_sulfur_dioxide'])
        density             =  float(request.form['density'])
        pH                  =  float(request.form['pH'])
        sulphates           =  float(request.form['sulphates'])
        alcohol             =  float(request.form['alcohol'])

            
        features = ['type', 'fixed acidity', 'volatile acidity', 'citric acid','residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        x_test = pd.DataFrame([[type,fixed_acidity, volatile_acidity, citric_acid,  residual_sugar,  chlorides, free_sulfur_dioxide,  total_sulfur_dioxide,  density,  pH,  sulphates, alcohol]], columns = features)


        predicted_val       = gb_model.predict_proba(x_test[features])
        predicted_quality   = ['Bad Quality','Good Quality','Best Quality'][np.argmax(predicted_val)]
        image               = ['Bad_quality_wine.jpg','Good_quality_wine.jpg','Best_quality_wine.jpg'][np.argmax(predicted_val)]
        k = {'quality_prediction':0}
        return jsonify({                           
                           'predicted_quality'      :   predicted_quality,
                           'image'                  :   image
            })
    else:
        return jsonify(defalut_wines_values)







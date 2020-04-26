# https://dash.plot.ly/dash-core-components

import dash
# contains widgets that can be dropped into app
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle

# new imports
#pip install basilica

import basilica
import numpy as np
import pandas as pd
import re
from scipy import spatial



# get data
#wget https://raw.githubusercontent.com/MedCabinet/ML_Machine_Learning_Files/master/med1.csv
# turn data into dataframe
df = pd.read_csv('symptoms6_medcab3.csv')

# get pickled trained embeddings for med cultivars
#wget https://github.com/lineality/4.4_Build_files/raw/master/medembedv2.pkl
#unpickling file of embedded cultivar descriptions
unpickled_df_test = pd.read_pickle("./symptommedembedv6.pkl")


########### Initiate the app
# 'app' is required by heroku
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
# server name is specified in proc file
server = app.server
app.title='NLP_embeddings'

########### Set up the layout
# generates HTML code
app.layout = html.Div(children=[
    html.H1('Medicinal Cultivar Recommendations by Symptom or Disease'),
    # multi line single-Div
    html.Div([
        # sections have similar code but unique slider id
        # header
        html.H6('Enter a description of your symptom or disease below, and get recommendations from Over 2,300 cultivars.'),

        dcc.Textarea(
            id='textarea',
            placeholder='Please describe your treatment needs here & wait for new results...E.g. headaches, pain, anziety',
            value='',
            style={'width': '100%'}
        ),
        #added linebreak so no overlap on screen
        html.Br(),
        # header
        # where output data will go
        html.H6(id='output-message', children='output will go here')
    ]),

    html.Br(),

    html.Br(),
    html.A('Inspect and Get The Open Source Code On Github', href='https://github.com/lineality/medflask1v1'),
    html.H6('''For best results, please use terms like the following:
anorexia
anziety
appetite
appetite loss (in addition to just 'appetite')
arthritis
cachexia or wasting syndrome
cancer
depression
eating disorder
epilepsy
fibromyalgia (not on state listing)
glaucoma
headaches
HIV / AIDS
inflamation
insomnia
migraines
ms
Multiple Sclerosis
nausia
pain
ptsd
panic attack
Spinal Cord
spasticity
'''),
html.Br(),
html.H6('E.g. pain, ms, appetite, spasticity'),
html.Br(),
html.H6('The prediction is made by calculating the Cosine Distance of the user text and the products https://reference.wolfram.com/language/ref/CosineDistance.html'),
html.Br(),
html.H6('Note: some of the cultivar names are themselves numbers. E.g. "1024" '),
html.Br(),
html.H6('See spreadsheet of data here to browse: https://docs.google.com/spreadsheets/d/16zUAlMi1qGugUGkVlONx9UKDtL9GTNeTX09Yn9l5mN4/edit?usp=sharing'),



])
############ Interactive Callbacks
# call back function, functions with decorators(specify input and output)
@app.callback(Output('output-message', 'children'),
            [Input('textarea', 'value')
            ])




#
def display_results(user_input):
    # this opens the pickle
    # the opposite of pickling the file

    # file = open(f'resources/model_k{k}.pkl', 'rb')
    # model=pickle.load(file)
    # file.close
    # new_obs=[[value0,value1]]
    # pred=model.predict(new_obs)
    # specieslist=['setosa', 'versicolor','verginica']
    # final_pred=specieslist[pred[0]]



    def predict(user_input):

        # Part 1
        # a function to calculate_user_text_embedding
        # to save the embedding value in session memory
        user_input_embedding = 0

        def calculate_user_text_embedding(input, user_input_embedding):

            # setting a string of two sentences for the algo to compare
            sentences = [input]

            # calculating embedding for both user_entered_text and for features
            with basilica.Connection('36a370e3-becb-99f5-93a0-a92344e78eab') as c:
                user_input_embedding = list(c.embed_sentences(sentences))

            return user_input_embedding

        # run the function to save the embedding value in session memory
        user_input_embedding = calculate_user_text_embedding(user_input, user_input_embedding)




        # part 2
        score = 0

        def score_user_input_from_stored_embedding_from_stored_values(input, score, row1, user_input_embedding):

            # obtains pre-calculated values from a pickled dataframe of arrays
            embedding_stored = unpickled_df_test.loc[row1, 0]

            # calculates the similarity of user_text vs. product description
            score = 1 - spatial.distance.cosine(embedding_stored, user_input_embedding)

            # returns a variable that can be used outside of the function
            return score



        # Part 3
        for i in range(2351):
            # calls the function to set the value of 'score'
            # which is the score of the user input
            score = score_user_input_from_stored_embedding_from_stored_values(user_input, score, i, user_input_embedding)

            #stores the score in the dataframe
            df.loc[i,'score'] = score


        # Part 4: returns all data for the top 5 results as a json obj
        df_big_json = df.sort_values(by='score', ascending=False)
        df_big_json = df_big_json.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1)
        df_big_json = df_big_json[:5]
        #df_big_json = df_big_json.to_json(orient='columns')
        string1 = df_big_json['Strain'].values
        string1 = str(string1)
        string1

        # Part 5: output
        return string1

    return f'Your top five recommended/predicted species/cultivars/strains, preceded by their search ranking scores, are "{predict(user_input)}"'

############ Execute the app
if __name__ == '__main__':
    app.run_server()

'''
To update github in terminal:
    $ git add .
    $ git status
    $ git commit -m 'Informative comment about what you did'
    $ git push origin master
'''

'''
To update heroku in terminal:
    $ heroku login
    $ git add .
    $ git commit -am "make it better"
    $ git push heroku master
'''

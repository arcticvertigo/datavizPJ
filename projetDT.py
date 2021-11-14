import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from functools import wraps
from matplotlib.pyplot import pie, axis, show
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import altair as alt

st.title("-- Study of the french dataset : Demandes de valeurs foncières --")


dfOption = pd.DataFrame({
'first column': ["2016",'2017','2018','2019',"2020"],
})

 
option = st.sidebar.selectbox('Which year would you like to study?',dfOption['first column'])

if option == "2016":
    st.subheader('year of study: 2016')
    filename = 'https://jtellier.fr/DataViz/full_2016.csv'
    DATE_COLUMN = "date_mutation"
elif option == '2017':
    st.subheader('year of study: 2017')
    filename = 'https://jtellier.fr/DataViz/full_2017.csv'
    DATE_COLUMN = "date_mutation"
elif option == '2018':
    st.subheader('year of study: 2018')
    filename = 'https://jtellier.fr/DataViz/full_2018.csv'
    DATE_COLUMN = "date_mutation"
elif option == '2019':
    st.subheader('year of study: 2019')
    filename = 'https://jtellier.fr/DataViz/full_2019.csv'
    DATE_COLUMN = "date_mutation"
elif option == '2020':
    st.subheader('year of study: 2020')
    filename = 'https://jtellier.fr/DataViz/full_2020.csv'
    DATE_COLUMN = "date_mutation"

user_choice_nbrows = st.sidebar.number_input('How many rows from the database do you want to study ? from 2 to 1000000', 10, 100000, 10000)
#user_choice_nbrows = st.sidebar.slider('How many rows from the database do you want to study ? from 10 to 1000000', 10, 100000, 10000)

@st.cache(allow_output_mutation=True)
def load_data(nrows):

    req_col=['date_mutation','nature_mutation','valeur_fonciere','code_postal','nom_commune','type_local','surface_reelle_bati','nombre_pieces_principales','nature_culture','surface_terrain','longitude','latitude']

    data=pd.read_csv(filename,nrows=user_choice_nbrows,usecols=req_col,parse_dates=['date_mutation'],dtype={("nature_mutation ","nom_commune","nature_culture"):"category",("valeur_fonciere","surface_relle_bati","nombre_pieces_principales","surface_terrain","longitude","latitude","code_postal") : "float32"})
    data['date_mutation']=pd.to_datetime(data["date_mutation"])
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)

    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text(" ")
if st.checkbox('Show raw data'):
    st.write(data)


#decorator to save time taken to execute a function
def log_time(func):
    def wrapper(*args,**kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        timeTaken = end - start

        d = open("timeoflog.txt", 'a')
        d.write("It took "+ str(timeTaken)+" seconds \n\n")

        d.close()
    return wrapper


def count_rows(rows): 
    return len(rows)


#map
@log_time
def mapping_data(dataset,col):
    d = open("timeoflog.txt", 'a')
    
    d.write("Map function\n")

    d.close()

    month_to_filter = st.slider('month', 1, 12, 2)
    filtered_data = dataset[dataset[col].dt.month == month_to_filter]
    st.subheader('Map des valeurs foncières le %seme mois' % month_to_filter)
    st.map(filtered_data, zoom=None) #mettre le zoom


#takes the dataframe and returns a dataframe with longitude, latitude and its date of mutation without null values (with which we can't use st.map())
def filterLonLatForMapping(df): 
    dt= [df["latitude"], df["longitude"], df["date_mutation"]]

    headers = ["latitude", "longitude", "date_mutation"]

    datanew1 = pd.concat(dt, axis=1, keys=headers)

    datanew = datanew1.dropna()
    return datanew




#display in table the nature of the locals and their minimaxmum values

def disp_table(dat):
    st.subheader('Table showing each nature mutation category and the sum of their loan values')

    data2 = dat.groupby('nature_mutation').agg({'valeur_fonciere': ['min','max','sum']})

    data2=data2.reset_index()

    data2.columns = ['nature_mutation','valeur_fonciere_min', 'valeur_fonciere_max', 'valeur_fonciere_sum']

    data2=data2.sort_values(by=['valeur_fonciere_sum'], ascending=False)
    fig = go.Figure(data=go.Table(header=dict(values=list(data2[["nature_mutation",'valeur_fonciere_min', 'valeur_fonciere_max', 'valeur_fonciere_sum']].columns), align='center', fill_color='#e2ffde'), cells=dict(values=[data2.nature_mutation,data2.valeur_fonciere_min,data2.valeur_fonciere_max,data2.valeur_fonciere_sum], align='left', fill_color='#fffede')))

    fig.update_layout(height=180,margin=dict(l=5,r=5,b=10,t=10))

    st.write(fig) 



#piechart display of communes sorted by the sum of their valeurs_foncière

def pie_chart_graph(dat):
    st.subheader('Pie chart of the communes according to the loan value')

    daf = dat.groupby('nom_commune').agg({'valeur_fonciere': 'sum'}).reset_index()
    dfdecroiss=daf.sort_values(by=['valeur_fonciere'], ascending=False)
    dfcroiss=daf.sort_values(by=['valeur_fonciere'], ascending=True)

    user_choice_pie = st.number_input('How many locations do you want to see?', 15)
    user_choice_pie=int(user_choice_pie)
    dfdecroiss=dfdecroiss.head(user_choice_pie)
    dfcroiss=dfcroiss.head(user_choice_pie)

    if st.checkbox('Lowest // Highest values'):
        st.markdown('%s first communes with the LOWEST land values' % user_choice_pie)
        
        fig = px.pie(dfcroiss, values='valeur_fonciere', names='nom_commune', hover_name='nom_commune')

        st.write(fig)
    else:
        st.markdown('%s first communes with the HIGHEST land values' % user_choice_pie)
        
        fig = px.pie(dfdecroiss, values='valeur_fonciere', names='nom_commune', hover_name='nom_commune')


        st.write(fig)


    st.markdown('%s first communes with the highest transactions' % user_choice_pie)
     
    df = dat.groupby(['nom_commune']).size().reset_index(name='number_transaction')
    df=df.head(user_choice_pie)

       
    fig = px.pie(df, values='number_transaction', names='nom_commune', hover_name='nom_commune')

    st.write(fig)
     
    user_choice_town = st.text_input('Enter a town name to see its specific number of transactions', 'Attignat')
    displayNbTown = df.loc[df['nom_commune'].isin([user_choice_town])]
    st.text('Number of transactions in '+ user_choice_town+':')
    st.text(displayNbTown['number_transaction'].values[0])


#line chart

def lines_chart_graph(dat):
    st.subheader('Plotly line chart showing the loan values of the towns per month. Do not hesitate to zoom in :')
    line_chart_data=dat.copy()

    line_chart_data['mutation_month'] = line_chart_data['date_mutation'].dt.month

    hour_cross_tab = pd.crosstab(line_chart_data['mutation_month'], line_chart_data['nom_commune'])

    fig = px.line(hour_cross_tab)

    fig.update_xaxes(type='category')

    st.write(fig)



def st_line(df):
    st.subheader('Line chart concerning the land surface of the goods.')
    df = df[['surface_terrain']]
    st.line_chart(df)




def scatter_plot(df):
    xcolumns_choice = list(df.select_dtypes(['float','int']).columns)

    df['month']=df['date_mutation'].dt.month

    columns_choice = list(df[['month','valeur_fonciere','code_postal','surface_terrain','nombre_pieces_principales','surface_reelle_bati']])
    

    st.subheader('- scatterplot: choose the X and Y axis parameters -')
    x_values = st.selectbox('X axis', options=xcolumns_choice)
    y_values = st.selectbox('Y axis', options=columns_choice)

    plot = px.scatter(data_frame=df, x=x_values,y=y_values)
    st.plotly_chart(plot)



def histogram_month(df):
    df.rename(columns={'surface_reelle_bati': 'surface batie', 'nombre_pieces_principales': 'nb pièces', 'surface_terrain': 'surface terrain'})

    st.subheader('Histrogram of the months')
    hist_values_month = np.histogram(df[DATE_COLUMN].dt.month, bins=12, range=(1,12))[0]
    st.bar_chart(hist_values_month)

    st.subheader('Histrogram of the postal codes and the number of principal rooms:')
    df = pd.DataFrame(df, columns = ['code_postal', 'nombre_pieces_principales'])
    df.hist()
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()




def heat_map(df):
    #log time file
    d = open("timeoflog.txt", 'a')
    d.write("Heatmap function\n")
    d.close()


    df['month']= df['date_mutation'].dt.month

    df['code_postal'] = df['code_postal'].round(decimals = -2)

    v1 = data.groupby(['code_postal','month']).apply(count_rows).unstack()

    st.subheader('Heatmap according to the months and postal code')

    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(v1, center=0, annot=True, ax=ax)
    st.write(fig)





def altair_chart(df):
    st.subheader('Altair chart of the loan surface according the land values:')
    c = alt.Chart(df).mark_circle().encode(
    x='valeur_fonciere', y='surface_terrain')

    st.altair_chart(c, use_container_width=True)





def bar_chart_highest_sold_area(daf):
    st.subheader('Bar chart of the towns with the highest areas of loan sold this year:')
    v2 = daf.groupby('nom_commune').agg({'surface_terrain': 'sum'}).reset_index()
    v2=v2.sort_values(by=['surface_terrain'], ascending=False)

    df = v2.head(15)

    chart = go.Figure(
        data=[go.Bar(x=df['nom_commune'], y=df['surface_terrain'], text = df['surface_terrain'],textposition = 'auto')])
    
    st.plotly_chart(chart, use_container_width=True)


def town_price_meters2(df):
    st.subheader('Towns along with their land values and land surface sold in the year')
    fig = px.scatter_3d(df, x='nom_commune', y='valeur_fonciere', z='surface_terrain')
    st.write(fig)



#-----------------------------------------------------MAIN----------------------------------------------------------------------

def main():

    mapping_data(filterLonLatForMapping(data), "date_mutation") #we map the data with the function mapping data, with parameters the function filterLonLatForMapping that takes the dataframe and return longitude and latitude without null values (with which we can't use st.map()) and our column date of mutation that will be used to choose the month we want to display on the map.
    disp_table(data)
    pie_chart_graph(data)
    town_price_meters2(data)
    lines_chart_graph(data)
    st_line(data)

    scatter_plot(data)
    histogram_month(data)
    heat_map(data)
    altair_chart(data)
    bar_chart_highest_sold_area(data)



if __name__ == "__main__":
    main()





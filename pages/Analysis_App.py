import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go 


def top_5_societies(df,dense):
    
    # means first need to find value_counts by societies and sort in descending order
    # Calculate the value counts for each society
    society_counts = df['society'].value_counts()

    # Select the top 5 most frequent societies
    if dense=='Highly Dense':
        societies = society_counts.head(5)
    else:
        societies = society_counts.tail(5)
        
       
    
    # Create the bar plot using Plotly Express for more simplicity and color options
    fig = px.bar(
        societies,
        x=societies.index,
        y=societies.values,
        color=societies.index,
        text='count',
        title=f'Top 5 {dense} Societies',
        labels={'count': 'Count', 'society': 'Society'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # Enhance the plot with more interactivity
    fig.update_traces(
        texttemplate='%{text:.2s}', textposition='outside'
    )
    fig.update_layout(
        uniformtext_minsize=8, uniformtext_mode='hide',
        xaxis_title='Society',
        yaxis_title='Count',
        legend_title='Society',
        yaxis=dict(showgrid=True),
        xaxis=dict(showgrid=True),
        template='plotly_dark'
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig)
    
def geo_map_analysis(df):
    
    latlong = pd.read_csv('data/raw/latlong.csv')
    
    latlong['lat']=latlong['coordinates'].str.split(',').str.get(0).str.split('°').str.get(0).astype('float')
    latlong['long']=latlong['coordinates'].str.split(',').str.get(1).str.split('°').str.get(0).astype('float')
    
    new_df = df.merge(latlong,on='sector')
    
    group = new_df.groupby(['sector'])[['price','price_per_sqft','built_up_area','lat','long']].mean()
    
    
    fig = px.scatter_mapbox(group, lat="lat", lon="long", color="price_per_sqft", size="built_up_area",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=10,
                  mapbox_style='open-street-map',text=group.index)
    st.plotly_chart(fig)
    


if __name__=='__main__':
    
    st.set_page_config(page_title="Plotting Demo")

    st.title('Analytics')

    df = pd.read_csv('prepared_data/missing_value_imputed.csv')
    
    st.header("Top 5 Societies by amount of properties these Societies Consists of: ")
    # Create a selectbox for the user to choose the type of visualization
    option = st.selectbox("Select Density Type:", ("Highly Dense", "Less Dense"))

    top_5_societies(df[df['society']!='independent'],option)
    
    st.header("Geographical Analyis of Gurgaon Properties")
    geo_map_analysis(df=df)
    

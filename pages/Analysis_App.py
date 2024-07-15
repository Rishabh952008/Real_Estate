import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go 
import ast 
import numpy as np


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
    
    
    fig = px.scatter_mapbox(group, lat="lat", lon="long",
                            color="price_per_sqft", size="built_up_area",
                  color_continuous_scale=px.colors.sequential.Viridis, 
                  size_max=20, zoom=10,
                  mapbox_style='open-street-map',
                  text=group.index,)
    st.plotly_chart(fig)
 

def wordcloud_build(colname,sector_name):
    wordcloud_df = pd.read_csv('pages/wordcloud_df.csv')
    use_df = wordcloud_df[[colname,'sector']]
    main =[]
    use_df =use_df[use_df['sector']==sector_name]
    for col in use_df[colname].dropna().apply(ast.literal_eval):
       main.extend(col)
    feature_text = ' '.join(main)
    
    # Create the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=set(['s'])).generate(feature_text)

    # Save the word cloud as an image
    wordcloud_image = wordcloud.to_image()

    # Convert the PIL image to a numpy array
    wordcloud_array = np.array(wordcloud_image)

    # Create a Plotly figure
    fig = px.imshow(wordcloud_array)
    fig.update_layout(coloraxis_showscale=False)  # Hide the color scale

    # Display the word cloud
    st.plotly_chart(fig)

def area_with_price(df,sector_name,property_type):
    
    
    # Create the scatter plot
    fig = px.scatter(df[(df['sector']==sector_name) & (df['property_type']==property_type)], x='built_up_area', y='price',
                     color='bedRoom',
                     color_continuous_scale=px.colors.sequential.Plasma,
                     hover_data={'luxury_score': True},
                     title='Scatter Plot of Prices vs. Area', 
                     labels={'area': 'Area (sq ft)', 'prices': 'Prices (Cr)'})

    # Show the plot
    st.plotly_chart(fig)
      
def pie_chart(df,colname,sector_name):
    fig = px.pie(df[df['sector']==sector_name],
                 names=colname,
                 title=f'Variation in %ge of {colname} in {sector_name}')
    st.plotly_chart(fig)

def boxplot(df):
    
    col = ['bedRoom','property_type','agePossession','sector']
    for x in col:
        
        fig = px.box(
    df,
    x=x,
    y='price',
    title=f'Box Plot of Price vs. {x}',
    labels={f'{x}': f'Number of {x}', 'price': 'Price (Cr)'},
    color=x) # Add color to distinguish between different bedroom counts
           # Show the plot in Streamlit
        st.plotly_chart(fig)
        
def distplot(df):
    color_palette = px.colors.qualitative.Alphabet
    col = ['bedRoom', 'property_type', 'agePossession', 'sector']
    for x in col:
        fig = px.histogram(
            df,
            x='price',
            color=x,
            marginal="rug",
            title=f'Distribution Plot of Price by {x}',
            labels={x: x.replace('_', ' ').title(), 'price': 'Price (Cr)'},
            # Adds a box plot to the histogram
            nbins=50,  # Number of bins in the histogram
            opacity=0.7,  # Transparency of the bars
            color_discrete_sequence=color_palette,
            histnorm='probability density'
        )
        st.plotly_chart(fig)


def sns_distplot(df):
    col = ['bedRoom', 'property_type', 'agePossession']
    
    for x in col:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df,
            x='price',
            hue=x,
            kde=True,  # Add the KDE (Kernel Density Estimate)
            palette='Set2',  # Use a vibrant color palette
            bins=50
        )
        plt.title(f'Distribution Plot of Price by {x}')
        plt.xlabel('Price (Cr)')
        plt.ylabel('Density')
        st.pyplot(plt)  # Display the plot in Streamlit
        plt.clf()  # Clear the figure to avoid overlap of plots


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
    
    st.header("Analysis of Facilities/nearbyLocation/description/furnishDetails/features/rating available in different sector using Word Cloud")
    columns = [
 'nearbyLocations',
 'furnishDetails',
 'features',
 'rating']
    
    # sector
    sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))
    Facilities = st.selectbox('Facilities',columns)
    wordcloud_build(sector_name=sector,colname=Facilities)
    
    #Scatter plot
    st.header('Scatter Plot for analysis of house prices variation according to area sector wise')
    Sector_Name = st.selectbox('Sector Name',sorted(df['sector'].unique().tolist()))
    property_type = st.selectbox('Property Type',['flat','house'])
    area_with_price(df=df,sector_name=Sector_Name,property_type=property_type)
    
    #overall scatter chart 
    st.header('Price vs Area including all sector for type of property you have choosen in Gurgaon ')
    st.plotly_chart(px.scatter(df[df['property_type']==property_type], x='built_up_area', y='price',
                               color='luxury_score',
                               color_continuous_scale=px.colors.sequential.Viridis,
                               title='Scatter Plot of Prices vs. Area', 
                               hover_data={'bedRoom': True},
                               labels={'area': 'Area (sq ft)', 'prices': 'Prices (Cr)'}))
    
    #Pie Chart
    st.header('Pie Chart Analysis')
    SectorName = st.selectbox('SectorName',sorted(df['sector'].unique().tolist()))
    columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    Choose_Category = st.selectbox('Cat',columns)
    pie_chart(df=df,colname=Choose_Category,sector_name=SectorName)
    
    #Box Plot
    st.header('Box Plots')
    boxplot(df=df)
    
    #Dist Plot
    st.header("Dist Plot")
    distplot(df=df)
    
    #Seaborn Distplot
    st.header("Seaborn Dist Plot")
    sns_distplot(df=df)
    
    

    
    
    
    

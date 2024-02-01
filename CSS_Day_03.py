# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:33:18 2024

@author: HadrianBezuidenhout
"""

# data-to-viz.com - Provides guidelines for plotting & code snippets

##############
# Line Plots
##############
# Matplotlib
#import matplotlib.pyplot as plt

# =============================================================================
# x_line = [1, 2, 3, 4, 5];
# y_line = [2, 4, 6, 8, 10];
# 
# 
# plt.plot(x_line, y_line, '-o')
# plt.xlabel("x_line")
# =============================================================================
#plt.ylabel("y_line")

#plt.title('Line Plot')
#plt.show()

# Plotly
#import plotly.express as px

# =============================================================================
# x_line = [1, 2, 3, 4, 5]
# y_line = [2, 4, 6, 8, 10]
# 
# fig = px.line(x=x_line, y=y_line, labels={'x': 'X-axis', 'y': 'Y-axis'}, title='Line Plot')
# fig.write_html("plot.html")
# =============================================================================

# This is used to automatically open up a browser of your plot
#import webbrowser
#webbrowser.open("plot.html")

############
# Bar Graph
############

# Matplotlib
# =============================================================================
# import matplotlib.pyplot as plt
# 
# x_bar = ['A', 'B', 'C', 'D']
# y_bar = [1, 2, 3, 4]
# 
# plt.bar(x_bar, y_bar)
# plt.xlabel('Categories')
# plt.ylabel('Values')
# plt.title('Bar Plot Example')
# plt.show()
# =============================================================================

# Plotly
# =============================================================================
# import plotly.express as px
# 
# x_bar = ['A', 'B', 'C', 'D']
# y_bar = [1, 2, 3, 4]
# fig = px.bar(x=x_bar, y=y_bar, labels={'x': 'Categories', 'y': 'Values'}, title='Bar Plot')
# fig.write_html("plot.html")
# 
# # This is used to automatically open up a browser of your plot
# import webbrowser
# webbrowser.open("plot.html")
# =============================================================================

################
# Scatter plot
################

# Matplotlib
# =============================================================================
# import matplotlib.pyplot as plt
# 
# x_scatter = [1, 2, 3, 4, 5]
# y_scatter = [2, 4, 6, 8, 10]
# 
# plt.scatter(x_scatter, y_scatter)
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Scatter Plot Example')
# plt.show()
# =============================================================================

# Plotly
# =============================================================================
# import plotly.express as px
# 
# x_scatter = [1, 2, 3, 4, 5]
# y_scatter = [2, 4, 6, 8, 10]
# 
# fig = px.scatter(x=x_scatter, y=y_scatter, labels={'x': 'X-axis', 'y': 'Y-axis'}, title='Scatter Plot')
# fig.write_html("plot.html")
# 
# # This is used to automatically open up a browser of your plot
# import webbrowser
# webbrowser.open("plot.html")
# =============================================================================

################
#Histogram Plot
################

# Matplotlib
# =============================================================================
# import matplotlib.pyplot as plt
# 
# x_histogram = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
# 
# plt.hist(x_histogram, bins=range(min(x_histogram), max(x_histogram) + 1), edgecolor='black')
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.title('Histogram Example')
# plt.show()
# =============================================================================

# Plotly
# =============================================================================
# import plotly.express as px
# 
# x_histogram = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
# 
# fig = px.histogram(x=x_histogram, labels={'x': 'Values'}, title='Histogram')
# fig.write_html("plot.html")
# 
# # This is used to automatically open up a browser of your plot
# import webbrowser
# webbrowser.open("plot.html")
# =============================================================================

###############
# Maps
###############
# =============================================================================
# import plotly.express as px
# data = px.data.gapminder()
# 
# # Create a choropleth world map
# fig = px.choropleth(
#     data_frame=data,
#     locations="iso_alpha",
#     color="gdpPercap",
#     hover_name="country",
#     animation_frame="year",
#     title="World Map Choropleth",
#     color_continuous_scale=px.colors.sequential.Plasma,
#     projection="natural earth"
# )
# fig.write_html("plot.html")
# 
# # This is used to automatically open up a browser of your plot
# import webbrowser
# webbrowser.open("plot.html")
# =============================================================================


##################
# Combining Plots
##################

# =============================================================================
# import plotly.express as px
# 
# df = px.data.gapminder().query("continent=='Oceania'")
# fig = px.line(df, x="year", y="lifeExp", color='country')
# fig.write_html("plot.html")
# 
# # This is used to automatically open up a browser of your plot
# import webbrowser
# webbrowser.open("plot.html")
# =============================================================================

# =============================================================================
# General EDA Guidelines:
# 1. Summary Statistics
# Use Pandas features like .info and .describe to get identfiy key varibales and dataset statistics. Check data types, nulls, and count for various columns.
# 
# 2. Data Visualization Techniques:
# Familiarize yourself with histograms, box plots, scatter plots, and correlation matrices using seaborn and matplotlib. Note insights revealed by different visualizations.
# 
# 3. Handling Missing Data and Outliers:
# Learn techniques for missing data handling and outlier identification using pandas. Note the impact of missing data on analysis.
# 
# 4. Univariate and Bivariate Analysis:
# Explore individual variable characteristics and relationships using descriptive statistics and visualizations. Analyze bivariate relationships through scatter plots, pair plots, and correlation analysis.
# 
# 5. Categorical Data Analysis:
# Analyze and visualize categorical data using bar charts, pie charts, and count plots. Note the significance of categorical data.
# 
# 6. Time Series Analysis:
# Analyze time series data using line plots, seasonal decomposition, and autocorrelation plots. Note challenges in time series analysis.
# =============================================================================

# Practice

import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# file = pd.read_csv("data/iris.csv")
# file['class'] = file['class'].str.replace('Iris-', '')
# 
# plt.plot(file["sepal_length"], file["sepal_width"])
# =============================================================================
# =============================================================================
# plt.xlabel("sepal length")
# plt.ylabel("sepal width")
# plt.show()
# =============================================================================
#import seaborn as sns
# =============================================================================
## Pair Plot
# sns.pairplot(file,hue="class")
# plt.show()
# =============================================================================

# =============================================================================
# # Pi Plot
# class_count = file['class'].value_counts()
# plt.pie(class_count, labels=class_count.index)
# plt.show()
# =============================================================================

# =============================================================================
# df = pd.read_csv("data/time_series_data.csv", index_col=0)
# df['Date']=pd.to_datetime(df['Date'], format = "%Y-%m-%d")
# print(df.info())
# 
# 
# df['Temperature'].plot(kind='hist',bins=20,title='Temperature')
# plt.show()
# =============================================================================

################
# Numpy
################
# x = np.array([1,2,3],[4,5,6],[7,8,9])
# x[0,:] # Slicing array - first row and all columns
# np.cross(x,y)
# np.matmul(x,y) # Matrix dot product
# x*y # Matrix element wise product
# d = np.linalg.det(x) # Determinant - if d>0 this matrix is not inderminant
# np.linalg.solve(a,b) # For two matrices a and b
# np.reshape() # Reshapes matrix

####################
# Curve Fitting
####################

import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt("Data/noisydata.csv",skiprows=1,delimiter=",") # same as read_csv except it returns a numpy array instead of a dataframe
data_avg = np.mean(data,0) # Average of data column 1
pressure = data[:,0]
flowrate = data[:,1]
fit = np.polyfit(pressure, flowrate, 2) # fits polynomial of second order
flowfit = np.polyval(fit,pressure)
plt.plot(pressure,flowrate,"go")
plt.plot(pressure,flowfit,"k-")
plt.xlabel("pressure (Pa)")
plt.ylabel("flow rate ($m^3/s$)")
plt.show()


#print(f"{x}") # putting an f before the the quotations to prit variables.





















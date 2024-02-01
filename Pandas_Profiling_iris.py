# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:22:13 2024

@author: HadrianBezuidenhout
"""

import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("Data/iris.csv")
profile = ProfileReport(df, title="Profiling Report")
profile.to_file("your_report.html")

import webbrowser
webbrowser.open("your_report.html")






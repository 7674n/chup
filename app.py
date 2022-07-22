import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression


data= pd.read_csv("salary.csv")
x=np.array(data['YearsExperience']).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data['Salary']))

st.title('Salary Predictor')
nav=st.sidebar.radio("Navigation",["Home","Prediction"])
if nav=="Home":
    st.image('salary.jpg')
    if st.checkbox("Show Table"):
        st.table(data)

    graph= st.selectbox("What kind of graph ? ",["Non-Interactive","Intractive"])
    val=st.slider("'YearsExperience'",0,25)
    data=data.loc[data['YearsExperience']>=val]
    if graph=="Non-Interactive":
        plt.figure(figsize=(15,10))
        figr, ax = plt.subplots()
        ax.scatter(data['YearsExperience'],data['Salary'])
        plt.ylim(0)
        plt.xlabel('YearsExperience')
        plt.ylabel('Salary')
        plt.tight_layout()
        st.pyplot(figr)
        
    if graph=="Intractive":
        layout =go.Layout(
            xaxis=dict(range=[0,16]),
            yaxis=dict(range=[0,170000])
        )
        fig = go.Figure(data=go.Scatter(x=data["YearsExperience"], y=data["Salary"], mode='markers'),layout = layout)
        st.plotly_chart(fig)
if nav=="Prediction":
    st.header("What is your Salary ?")
    val=st.number_input('Enter your experience',step=0.25)
    val=np.array(val).reshape(1,-1)
    pred= lr.predict(val)[0]
    if st.button("Predict"):
        st.success(f"Your predicted salary is {round(pred)}")
# if nav=="Contribute":
#     st.header("Contribute to dataset")
#     ex= st.number_input('Enter your experience',0.00,20.00,step=.25)
#     sal=st.number_input('Enter your salary ',0.00,100000.00, step=1000.00)
#     age=st.number_input('Enter your Age',25.0,45.0, step=0.25)
#     Dist=st.number_input('Enter your distance',25.0,150.0, step=0.25)


#     if st.button("Submit"):
#         to_add = {'age':[age],'distance':[Dist],"YearsExperience":[ex],"Salary":[sal]}
#         to_add = pd.DataFrame(to_add)
#         to_add.to_csv("salary.csv",mode='a',header = False,index= False)
#         st.success("Submitted")

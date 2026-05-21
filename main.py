#Step1
#importing the Libraries
#====================================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_community.llms import Ollama

#========================================================================
#step2
#page Configuration 
#==========================================================================
st.set_page_config(page_title = "AI Data Analyst",layout = "wide")

#Step3
#Application Tittle
#============================================================================
st.title("📊 AI Data Analyst Agent")
st.markdown("Upload a CSV file and let AI analyze your dataset automatically.")

#step4
#Sidebar
#============================================================================
with st.sidebar:
     st.header("⚙️ Configuration")
     st.write("This Project USes:")

     st.write("✅ Streamlit")
     st.write("✅ Ollama")
     st.write("✅ Mistral AI")
     st.write("✅ Pandas")
     st.write("✅ Seaborn")

#step5
#File Uploader
#==================================================================================
uploaded_file = st.file_uploader("Choose the CSV File",type = ["CSV"])

#step6
#If File is Uploaded
#==================================================================================
if uploaded_file:
    df = pd.read_csv(uploaded_file) #Read CSV File
    
#Display DataFrame
# ===================================================================================    
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

#step7
# Dataset Schema
# ======================================================================================
    schema_info = {"columns": df.columns.tolist(),
                   "dtypes": df.dtypes.astype(str).to_dict(),
                   "missing_values": df.isnull().sum().to_dict(),
                   "shape": df.shape
                  } 

#step8
# Display Schema
# ============================================================================================
    st.subheader("📌 Dataset Schema")
    st.write(schema_info)


#step9
#Initializing Ollama Model
#==============================================================================================
    llm = Ollama(model = "mistral")

#Button for AI Analysis
    if st.button("🚀 Generate Analysis Code & Insights"):
#spinner
        with st.spinner("Writing EDA code and generating insights..."):
             
#prompt Engineering             
            prompt = f"""
            You are an expert Data Scientist.

            Analyze the uploaded dataset.

            Dataset Schema:
            {schema_info}

            Generate:

            1. Key insights
            2. Business observations
            3. Data quality issues
            4. Recommendations
            5. Important trends

            Give the response in simple bullet points.
            """
            
# AI Response            
            response = llm.invoke(prompt)
#Display AI Response
            st.subheader("🤖 AI Generated Insights")
            st.markdown(response)

#Statistical Summary
    st.subheader("📈 Statistical Summary")
    st.write(df.describe(include="all"))

#Data Visualization
    st.subheader("📊 Data Visualization")

#Select Numeric Columns
    numeric_columns=df.select_dtypes(include=['int64','float64']).columns

#Histogram        
    if len(numeric_columns) > 0:

        selected_column = st.selectbox(
            "Select Numeric Column",
            numeric_columns
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.histplot(df[selected_column], kde=True, ax=ax)

        ax.set_title(f"Distribution of {selected_column}")

        st.pyplot(fig)

#corelation Heatmap
    if len(numeric_columns) > 1:

        st.subheader("🔥 Correlation Heatmap")

        fig2, ax2 = plt.subplots(figsize=(8, 5))

        correlation = df[numeric_columns].corr()

        sns.heatmap(
            correlation,
            annot=True,
            cmap='coolwarm',
            ax=ax2
        )

        st.pyplot(fig2)
#If no File is Uploaded
    else:

     st.info("📁 Please upload a CSV file to begin analysis.")                     

        


import streamlit as st
import pandas as pd
import io

# Title of the Streamlit app
st.title('Data Filter & Export App')

# Section to upload the file
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_table(uploaded_file)
    
    # Display original data and its shape
    st.write("Original Data:")
    st.dataframe(df)
    st.write("Original Shape:", df.shape)
    
    # Identify columns that start with "LFQ"
    lfq_columns = [col for col in df.columns if col.startswith('LFQ')]
    
    # Slider for selecting the percentage
    percentage = st.slider('Minimum percentage of LFQ columns that are not 0:', 0, 100, 60)
    
    # Calculate and filter the DataFrame
    df['lfq_valid_percentage'] = df[lfq_columns].apply(lambda x: (x != 0).mean(), axis=1) * 100
    filtered_df = df[df['lfq_valid_percentage'] >= percentage].drop(columns=['lfq_valid_percentage'])
    
    # Display filtered data and its shape
    st.write("Filtered Data:")
    st.dataframe(filtered_df)
    st.write("Filtered Shape:", filtered_df.shape)

    # Function to convert DataFrame to CSV (bytes)
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(sep='\t', index=False).encode('utf-8')

    csv = convert_df_to_csv(filtered_df)
    
    # Download button for the filtered DataFrame as text file
    st.download_button(
        label="Download data as TXT",
        data=csv,
        file_name='filtered_data.txt',
        mime='text/csv',
    )
    


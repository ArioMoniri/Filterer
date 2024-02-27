import streamlit as st
import pandas as pd
import io

# Initialize session state for tracking user's decision on contaminated proteins
if 'delete_contaminated' not in st.session_state:
    st.session_state.delete_contaminated = None


# Title of the Streamlit app
st.title('Data Filterer')

# Function to filter out contaminated proteins
def filter_contaminated_proteins(df):
    return df[~df['Protein IDs'].str.contains("CON")]

# Function to filter based on protein ID prefixes
def filter_prefixes(df):
    # Extract prefixes
    df['Prefix'] = df['Protein IDs'].apply(lambda x: x.split("|")[0])
    unique_prefixes = df['Prefix'].unique()

    # Display checkboxes for each prefix and collect user selections
    st.write("Select the protein types you want to exclude:")
    selections = {prefix: st.checkbox(prefix, True) for prefix in unique_prefixes}

    # Filter out unselected prefixes
    for prefix, selected in selections.items():
        if not selected:
            df = df[~df['Prefix'].str.contains(prefix)]
    
    # Drop the temporary 'Prefix' column
    df.drop(columns=['Prefix'], inplace=True)
    
    return df

# Section to upload the file
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_table(uploaded_file)
    
    # Display original data and its shape
    st.write("Original Data:", df.shape)
    st.dataframe(df)
    #st.write("Original Shape:", df.shape)

    # Ask if the user wants to delete contaminated proteins
    if st.checkbox("Do you want to delete contaminated proteins?", key='delete_check'):
        df = filter_contaminated_proteins(df)
        st.session_state.delete_contaminated = True
        st.success("Contaminated proteins deleted successfully.")
        st.write("Filtered Data:")
        st.dataframe(df)
        st.write("Filtered Shape:", df.shape)
    elif 'delete_check' in st.session_state:
        st.session_state.delete_contaminated = False

    # Conditionally display the rest of the app based on the user's choice
    if st.session_state.delete_contaminated is not None:
        # Show unique protein types before filtering
        st.write("Protein types before filtering:")
        df['Prefix'] = df['Protein IDs'].apply(lambda x: x.split("|")[0])
        st.write(df['Prefix'].unique())
        df.drop(columns=['Prefix'], inplace=True)  # Clean up

        # Ask user for prefix filters after showing unique types
        df = filter_prefixes(df)
        
        # Display the DataFrame after filtering
        st.write("Data after filtering by prefixes:",df.shape)
        


    # Conditionally display the rest of the app based on the user's choice
    if st.session_state.delete_contaminated is not None:
        
        # Identify columns that start with "LFQ"
        lfq_columns = [col for col in df.columns if col.startswith('LFQ')]
        
        # Continue with your analysis or display the DataFrame
        st.write(df)
        # Slider for selecting the percentage
        percentage = st.slider('Minimum percentage of LFQ columns that are not 0:', 0, 100, 50)
        
        # Calculate and filter the DataFrame
        df['lfq_valid_percentage'] = df[lfq_columns].apply(lambda x: (x != 0).mean(), axis=1) * 100
        filtered_df = df[df['lfq_valid_percentage'] >= percentage].drop(columns=['lfq_valid_percentage'])
        
        # Display filtered data and its shape
        st.write("Filtered Data:",filtered_df.shape)
        st.dataframe(filtered_df)
        #st.write("Filtered Shape:", filtered_df.shape)
    
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
    
     # Function to convert DataFrame to Excel (bytes)
        @st.cache_data
        def convert_df_to_excel(df):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            return output.getvalue()
        
        excel = convert_df_to_excel(filtered_df)
        
        # Download button for the filtered DataFrame as Excel file
        st.download_button(
            label="Download data as Excel",
            data=excel,
            file_name='filtered_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
        
    import streamlit as st
    import pandas as pd
    import numpy as np
    from scipy import stats
    import io
    
    # Assuming df is your DataFrame after uploading and filtering
    
    # Function to perform ANOVA and t-tests
    def perform_statistical_tests(df):
        # Filter columns starting with "LFQ" and replace 0s with NaN
        lfq_columns = df.filter(regex='^LFQ').replace(0, np.nan)
        
        # Perform ANOVA
        anova_result = stats.f_oneway(*(lfq_columns[col].dropna() for col in lfq_columns))
        
        # Store results in a dictionary for easy access
        results = {'ANOVA': {'Statistic': anova_result.statistic, 'P-Value': anova_result.pvalue}}
        
        # Perform pairwise t-tests
        t_tests = {}
        for i, col1 in enumerate(lfq_columns.columns):
            for col2 in lfq_columns.columns[i+1:]:
                t_stat, p_val = stats.ttest_ind(lfq_columns[col1].dropna(), lfq_columns[col2].dropna(), nan_policy='omit')
                t_tests[f'{col1} vs {col2}'] = {'T-Statistic': t_stat, 'P-Value': p_val}
        
        results['T-Tests'] = t_tests
        return results
    
    # Function to convert results to a downloadable format
    @st.cache_data
    def convert_results_to_csv(results):
        output = io.StringIO()
        output.write("Test,Statistic,P-Value\n")
        output.write(f"ANOVA,{results['ANOVA']['Statistic']},{results['ANOVA']['P-Value']}\n")
        for test, values in results['T-Tests'].items():
            output.write(f"{test},{values['T-Statistic']},{values['P-Value']}\n")
        return output.getvalue().encode('utf-8')
    
    # Example usage in Streamlit after data is filtered and ready
    if st.button('Perform Statistical Tests'):
        results = perform_statistical_tests(filtered_df)  # Assume filtered_df is your filtered DataFrame
        st.write("ANOVA Test Result:", results['ANOVA'])
        st.write("Pairwise T-Test Results:")
        for test, values in results['T-Tests'].items():
            st.write(f"{test}: {values}")
        
        # Convert results to CSV for download
        csv_results = convert_results_to_csv(results)
        st.download_button(
            label="Download Test Results as CSV",
            data=csv_results,
            file_name='statistical_test_results.csv',
            mime='text/csv',
        )


        pass

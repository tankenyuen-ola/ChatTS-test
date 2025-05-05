batch_prompt = """
            I have {num_valid_series} time series datasets related to '{title}', with the following indicators: {valid_indicator_string} respectively. 
            The data series are provided here: {ts_placeholders}. 
            Please perform a comprehensive analysis that includes:\n
            1. Identifying local and global patterns, trends, and anomalies within each time series.\n
            2. Exploring correlations and potential causal relationships between the different indicators.\n
            3. Linking observed changes to plausible real-world scenarios or events, better related to {title}, that could have influenced these patterns.\n
            4. Excluding potential causes that are inconsistent with the observed data trends.\n
            Provide a detailed explanation of your findings, including any assumptions made and the reasoning behind your conclusions.
            """

interactive_prompt = """
            I have {num_valid_series} time series datasets related to '{title}', with the following indicators: {valid_indicator_string} respectively.
            The data series are provided here: {ts_placeholders}. 
            Here is the chat history between the user and the assistant that may help to answer the following question better. \n
            {chat_history} \n
            Here is the question that is asked by user: {question}. Please answer the question in detail based on the given data, and you may refer to the chat history if needed. \n
        """
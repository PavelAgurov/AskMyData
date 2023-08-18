"""
    Main
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305

import os
import pandas as pd

import streamlit as st

from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe
from pandasai.helpers.openai_info import get_openai_callback
from pandasai.middlewares import StreamlitMiddleware

from strings import HEADER_STR, HOW_IT_WORKS, GPT_KEY_USAGE_WARNING, UPLOAD_FILE_MESSAGE
from utils_streamlit import streamlit_hack_remove_top_space

# ------------------------------- UI

st.set_page_config(page_title= HEADER_STR, layout="wide")
st.title(HEADER_STR)

streamlit_hack_remove_top_space()

tab_main, tab_setting, tab_debug = st.tabs(["Request", "Settings", "Debug"])

with tab_main:
    header_container = st.container()
    uploaded_file = st.file_uploader(
        UPLOAD_FILE_MESSAGE,
        type="csv"
    )
    data_header = st.expander(label="First 5 rows").empty()
    question_container = st.empty()
    question = question_container.text_input("Enter your question and press Enter:", value="", type="default")
    result_container = st.empty()
    debug_container = st.container()

with tab_setting:
    gpt_key_info = st.info(GPT_KEY_USAGE_WARNING)
    open_api_key = st.text_input("OpenAPI Key: ", "", key="open_api_key")
    cb_conversational = st.checkbox(label="Conversational mode", value=True)

#with tab_debug:

with st.sidebar:
    token_count_container = st.empty()

header_container.markdown(HOW_IT_WORKS, unsafe_allow_html=True)

# ------------------------------- Session
if 'data' not in st.session_state:
    st.session_state.data = None

# ------------------------------- LLM

if open_api_key:
    LLM_OPENAI_API_KEY = open_api_key
else:
    LLM_OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = OpenAI(api_token= LLM_OPENAI_API_KEY)

# ------------------------------- App

# ask to upload file
if st.session_state.data is None:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df

# no data yet
if st.session_state.data is None:
    st.stop()

df = st.session_state.data
data_header.dataframe(df.head(), use_container_width=True, hide_index=True)

# we have data, but no question yet
if not question:
    st.stop()

# we have all - can run it
try:
    with get_openai_callback() as cb:
        smart_df = SmartDataframe(df, config={
                        "llm": llm, 
                        "conversational": cb_conversational, 
                        "enable_cache": True,
                        "middlewares": [StreamlitMiddleware()]
                    })
        result = smart_df.chat(question)

        debug_container.markdown(type(result))

        if isinstance(result, SmartDataframe):
            result_container.dataframe(result, use_container_width= True, hide_index= True)
        else:
            result_container.markdown(result)

except Exception as error: # pylint: disable=W0718,W0702
    debug_container.markdown(f'Error: {error}')


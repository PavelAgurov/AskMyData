"""
    Main
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305

import os
import pandas as pd
import traceback
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st

from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe
from pandasai.helpers.openai_info import get_openai_callback
from pandasai.middlewares import StreamlitMiddleware, ChartsMiddleware
from pandasai.helpers.logger import Logger

from strings import HEADER_STR, HOW_IT_WORKS, GPT_KEY_USAGE_WARNING, UPLOAD_FILE_MESSAGE, LOAD_TITANIK_DATA
from utils_streamlit import streamlit_hack_remove_top_space, streanlit_hide_main_menu

MODEL_NAME = "gpt-3.5-turbo" # gpt-3.5-turbo-16k
OUTPUT_GPAPH_FOLDER = './exports/charts/'
OUTPUT_GPAPH_FILE   = './exports/charts/temp_chart.png'
# ------------------------------- Functions

def show_used_tokens():
    """Show token counter"""
    token_count_container.markdown(f'Used {st.session_state.tokens} tokens')

def init_graph_folder():
    if not os.path.exists(OUTPUT_GPAPH_FOLDER):
        os.makedirs(OUTPUT_GPAPH_FOLDER)

def clear_graph_file():
    """Remove png file if exists"""
    if os.path.exists(OUTPUT_GPAPH_FILE):
        os.remove(OUTPUT_GPAPH_FILE)

# ------------------------------- UI

st.set_page_config(page_title= HEADER_STR, layout="wide")
st.title(HEADER_STR)

streamlit_hack_remove_top_space()
#streanlit_hide_main_menu()

tab_main, tab_setting, tab_debug = st.tabs(["Request", "Settings", "Debug"])

with tab_main:
    header_container = st.container()
    col1 , col2 = st.columns([6, 1])
    uploaded_file = col1.file_uploader(
        UPLOAD_FILE_MESSAGE,
        type=["csv", "xls", "xslx"],
        accept_multiple_files= False
    )
    load_titanik_msg = col2.markdown(LOAD_TITANIK_DATA, unsafe_allow_html=True)
    load_titanik_button = col2.button('Load')
    loading_status = st.empty()
    data_header = st.expander(label="First 5 rows", expanded=True).empty()
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
    result_type = st.empty()
    error_log = st.expander(label="Error log").empty()
    trace_log = st.expander(label="Trace log").empty()

header_container.markdown(HOW_IT_WORKS, unsafe_allow_html=True)

# ------------------------------- Session
if 'data' not in st.session_state:
    st.session_state.data = None
if 'tokens' not in st.session_state:
    st.session_state.tokens = 0

# ------------------------------- LLM

if open_api_key:
    LLM_OPENAI_API_KEY = open_api_key
else:
    LLM_OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = OpenAI(api_token= LLM_OPENAI_API_KEY, model = MODEL_NAME, temperature=0, max_tokens=1000)

# ------------------------------- App

show_used_tokens()
init_graph_folder()
clear_graph_file()

# ask to upload file
if uploaded_file is not None:
    file_name : str = uploaded_file.name
    if file_name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        df = pd.read_excel(uploaded_file)
    else:
        df = None
    st.session_state.data = df

if load_titanik_button:
    df = pd.read_csv('./data_examples/titanic.csv')
    st.session_state.data = df

# no data yet
if st.session_state.data is None:
    loading_status.markdown('Data is not loaded yet')
    result_container.markdown('')
    debug_container.markdown('')
    result_type.markdown('')
    trace_log.markdown('')
    st.stop()

df = st.session_state.data
loading_status.markdown(f'Loaded {df.shape} data.')
data_header.dataframe(df.head(), use_container_width=True, hide_index=True)

# we have data, but no question yet
if not question:
    st.stop()

# we have all - can run it
logger = Logger(verbose=True)
try:
    smart_df = SmartDataframe(df, config={
                    "llm": llm, 
                    "conversational": cb_conversational, 
                    "enable_cache": False,
                    "middlewares": [StreamlitMiddleware(), ChartsMiddleware()]
                    }, 
                    logger= logger,
                    
                )

    with get_openai_callback() as cb:
        result = smart_df.chat(question)
    st.session_state.tokens += cb.total_tokens
    show_used_tokens()

    result_type.markdown(type(result))

    if os.path.exists(OUTPUT_GPAPH_FILE):
        image = Image.open(OUTPUT_GPAPH_FILE)
        result_container.image(image)
    else:
        if isinstance(result, (SmartDataframe, pd.DataFrame, pd.Series)):
            result_container.dataframe(result, use_container_width= True, hide_index= True)
        else:
            result_container.markdown(result)

except Exception as error: # pylint: disable=W0718,W0702
    error_log.markdown(f'Error: {error}. Track: {traceback.format_exc()}')
    result_container.markdown("Sorry, I can't process your question. It's still POC. I hope this problem will be fixed soon.")

if logger.logs:
    trace_log.text('\n'.join([log_item['msg'] for log_item in logger.logs]))
else:
    trace_log.markdown('No logs yet')


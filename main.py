"""
    Main
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411

import os
import pandas as pd
import traceback
from PIL import Image

import streamlit as st

from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe
from pandasai.helpers.openai_info import get_openai_callback
from pandasai.helpers.logger import Logger

from strings import  WHAT_IS_TITANIK_DATA, QUESTION_EXAMPLES
from utils_streamlit import streamlit_hack_remove_top_space, streanlit_hide_main_menu

MODEL_NAME = "gpt-3.5-turbo" # gpt-3.5-turbo-16k
OUTPUT_GPAPH_FOLDER = './exports/charts/'
OUTPUT_GPAPH_FILE   = './exports/charts/temp_chart.png'

# ------------------------------- Functions

def show_used_tokens():
    """Show token counter"""
    token_count_container.markdown(f'Used {st.session_state.tokens} tokens')

def init_graph_folder():
    """Create ouput folder if doesn't exist"""
    if not os.path.isdir(OUTPUT_GPAPH_FOLDER):
        os.makedirs(OUTPUT_GPAPH_FOLDER)

def clear_graph_file():
    """Remove png file if exists"""
    if os.path.exists(OUTPUT_GPAPH_FILE):
        os.remove(OUTPUT_GPAPH_FILE)

# ------------------------------- UI

HEADER_STR = "Ask Your Data POC"
st.set_page_config(page_title= HEADER_STR, layout="wide")
st.title(HEADER_STR)

streamlit_hack_remove_top_space()
streanlit_hide_main_menu()

tab_main, tab_setting, tab_debug = st.tabs(["Request", "Settings", "Debug"])

with tab_main:
    header_container = st.container().markdown("Upload your file Csv or Excel and ask questions.", unsafe_allow_html=True)
    col11 , col21 = st.columns([6, 1])

    file_uploader_container = col11.container(border=True)
    uploaded_file = file_uploader_container.file_uploader(
        "Choose a data file (Csv).",
        type=["csv", "xls", "xslx"],
        accept_multiple_files= False
    )
    default_data_container = col21.container(border=True)
    default_data_container.markdown('Load data example:')
    load_titanik_button = default_data_container.button('Load Titanik data')
    default_data_container.markdown(WHAT_IS_TITANIK_DATA, unsafe_allow_html=True)

    loading_status = st.empty()
    data_header = st.expander(label="First 5 rows", expanded=True).empty()
    st.expander(label="Example of questions...").empty().markdown(QUESTION_EXAMPLES)
    st.empty()
    
    question = st.text_input("Your question:", value="", type="default")
    
    result_container = st.empty()
    debug_container = st.container()

with tab_setting:
    gpt_key_info = st.info("Enter your Gpt key and press Enter. Your key is only used in your browser. If you refresh the page - enter the key again.")
    open_api_key = st.text_input("OpenAPI Key: ", "", key="open_api_key")
    cb_conversational = st.checkbox(label="Conversational mode", value=True)
#    cb_enforce_privacy = st.checkbox(label="Do not send data to the API (will send only headers)", value=False)

with st.sidebar:
    token_count_container = st.empty()
    result_type = st.empty()
    error_log = st.expander(label="Error log").empty()
    trace_log = st.expander(label="Trace log").empty()

# ------------------------------- Session
if 'data' not in st.session_state:
    st.session_state.data = None
if 'tokens' not in st.session_state:
    st.session_state.tokens = 0

# ------------------------------- LLM

if open_api_key:
    os.environ["OPENAI_API_KEY"] = open_api_key
else:
    all_secrets = {s[0]:s[1] for s in st.secrets.items()}
    openai_secrets = all_secrets.get('open_api_openai')
    if openai_secrets:
        os.environ["OPENAI_API_KEY"] = openai_secrets.get('OPENAI_API_KEY')

LLM_OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
llm = OpenAI(api_token= LLM_OPENAI_API_KEY, model = MODEL_NAME, temperature=0, max_tokens=1000)

# ------------------------------- App

show_used_tokens()
init_graph_folder()
clear_graph_file()

# load default data
if load_titanik_button:
    df = pd.read_csv('./data_examples/titanic.csv')
    st.session_state.data = df
    st.stop()

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
    st.stop()

# no data yet
if st.session_state.data is None:
    loading_status.markdown('Data is not loaded yet')
    result_container.markdown('Error: data was not loaded. Load data from Excel/Csv or data example.')
    debug_container.markdown('')
    result_type.markdown('')
    trace_log.markdown('')
    st.stop()

df = st.session_state.data
loading_status.markdown(f'Loaded {df.shape} rows.')
data_header.dataframe(df.head(), use_container_width=True, hide_index=True)

# we have data, but no question yet
if not question:
    result_container.markdown('Enter you question and click Enter')
    st.stop()

# we have all - can run it
logger = Logger(verbose=True)
try:
    smart_df = SmartDataframe(df, config={
                    "llm": llm, 
                    "conversational": cb_conversational,
                    "enable_cache": True,
#                    "enforce_privacy" : cb_enforce_privacy
                    }, 
                    logger= logger,
                    
                )

    with st.spinner(text="In progress..."):
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


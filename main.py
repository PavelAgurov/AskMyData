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
from pandasai import SmartDataframe, SmartDatalake
from pandasai.responses import StreamlitResponse
from pandasai.helpers.openai_info import get_openai_callback
from pandasai.helpers.logger import Logger

from strings import  WHAT_IS_TITANIK_DATA, QUESTION_EXAMPLES
from utils_streamlit import streamlit_hack_remove_top_space, streanlit_hide_main_menu

MODEL_NAME = "gpt-3.5-turbo" # gpt-3.5-turbo-16k
OUTPUT_GPAPH_FOLDER = '.exports_charts$$'

DEBUG_GUID = 'f0bec9d3-1ec3-4bc0-a41e-dca19b9a6c9d'

# ------------------------------- Session
if 'data' not in st.session_state:
    st.session_state.data = []
    st.session_state.data_names = []
    st.session_state.data_example = None
if 'tokens' not in st.session_state:
    st.session_state.tokens = 0

# ------------------------------- Functions

def show_used_tokens():
    """Show token counter"""
    token_count_container.markdown(f'Used {st.session_state.tokens} tokens')

def init_graph_folder():
    """Create ouput folder if doesn't exist"""
    if not os.path.isdir(OUTPUT_GPAPH_FOLDER):
        os.makedirs(OUTPUT_GPAPH_FOLDER)

def get_data_shape_str(df_data):
    """Get data share"""
    if df_data is None:
        return "(not loaded)"
    return f"(loaded {df_data.shape})"

def load_data_file(uploaded_file):
    """Load one data file"""
    loaded_file_name = uploaded_file.name    
    if loaded_file_name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    if loaded_file_name.endswith('.xlsx') or loaded_file_name.endswith('.xls'):
        return pd.read_excel(uploaded_file)
    return None

def get_uploaded_data(uploaded_files):
    """Upload data files into DataFrame"""
    file_names = []
    for uploaded_file in uploaded_files:
        file_names.append(uploaded_file.name)

    result_data = []
    for uploaded_file in uploaded_files:
        result_data.append(load_data_file(uploaded_file))

    return result_data, file_names

# ------------------------------- UI

HEADER_STR = "Ask Your Data POC"
st.set_page_config(page_title= HEADER_STR, layout="wide")
st.title(HEADER_STR)

streamlit_hack_remove_top_space()
streanlit_hide_main_menu()

st.markdown(st.get_option('browser.serverAddress'))

tab_main, tab_setting, tab_debug = st.tabs(["Request", "Settings", "Debug"])

with tab_main:
    header_container = st.container().markdown("Upload your file Csv or Excel and ask questions.", unsafe_allow_html=True)

    main_data_container = st.empty().container(border=True)
    tab_load_data, tab_load_examples = main_data_container.tabs(["Load from file(s)", "Load data examples"])

    with tab_load_data:
        with st.form("my-form", clear_on_submit=True, border=False):
            uploaded_files = st.file_uploader(
                "Choose first data file (csv, xls, xslx).",
                type=["csv", "xls", "xslx"],
                accept_multiple_files= True
            )
            submitted_uploaded_files = st.form_submit_button("Upload selected files")
    
    with tab_load_examples:
        cols_examples = st.columns(3)
        load_titanik_button = cols_examples[0].button('Load Titanik data')
        load_country_button = cols_examples[1].button('Load Country data')
        load_dow2011_button = cols_examples[2].button('Load Dow Jones index 2011')
        st.markdown(WHAT_IS_TITANIK_DATA, unsafe_allow_html=True)

    data_headers = st.expander(label="First 5 data rows", expanded=True).empty()

    if st.session_state.data:
        data_tabs = data_headers.tabs(st.session_state.data_names)
        for df_index, df in enumerate(st.session_state.data):
            if df is not None:
                data_tabs[df_index].dataframe(df.head(), use_container_width=True, hide_index=True)

    question = st.text_input("Your question:", value="", type="default")
    
    result_container = st.empty()
    debug_container = st.container()

with tab_setting:
    gpt_key_info = st.info("Enter your Gpt key and press Enter. Your key is only used in your browser. If you refresh the page - enter the key again.")
    open_api_key = st.text_input("OpenAPI Key: ", "", key="open_api_key")
    cb_conversational = st.checkbox(label="Conversational mode", value=True)
#    cb_enforce_privacy = st.checkbox(label="Do not send data to the API (will send only headers)", value=False)

with tab_debug:
    debug_guid = st.text_input(label='DebugID', type='password')

with st.sidebar:
    st.expander(label="Example of questions...").empty().markdown(QUESTION_EXAMPLES)
    token_count_container = st.empty()
    result_type = st.empty()
    error_log = st.expander(label="Error log").empty()
    trace_log = st.expander(label="Trace log").empty()


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

if debug_guid == DEBUG_GUID:
    with open('pandasai.log', 'rb') as f:
       st.download_button('Download Log', f, file_name='pandasai.log')

# https://github.com/datasciencedojo/datasets/blob/master/titanic.csv
if load_titanik_button:
    st.session_state.data = [pd.read_csv('./data_examples/titanic.csv')]
    st.session_state.data_names = ['Titanic']
    st.session_state.data_example = 'Titanic'
    st.rerun()

# https://github.com/datasciencedojo/datasets/tree/master/WorldDBTables
if load_country_button:    
    st.session_state.data = [pd.read_csv('./data_examples/CityTable.csv'), pd.read_csv('./data_examples/CountryTable.csv'), pd.read_csv('./data_examples/LanguageTable.csv')]
    st.session_state.data_names = ['City', 'Country', 'Language']
    st.session_state.data_example = 'Cities'
    st.rerun()

# https://code.datasciencedojo.com/datasciencedojo/datasets/blob/master/Dow%20Jones%20Index/dow_jones_index.data
if load_dow2011_button:
    st.session_state.data = [pd.read_csv('./data_examples/dow_jones_index_2011.csv')]
    st.session_state.data_names = ['Dow Jones index 2011']
    st.session_state.data_example = 'Dow Jones index 2011'
    st.rerun()

# upload file
if submitted_uploaded_files:
    new_data, new_file_names = get_uploaded_data(uploaded_files)
    st.session_state.data = new_data
    st.session_state.data_names = new_file_names
    st.session_state.data_example = ''
    st.rerun()

# no data yet
if len(st.session_state.data) == 0:
    result_container.markdown('Data was not loaded. Load data from Excel/Csv or data example.')
    debug_container.markdown('')
    result_type.markdown('')
    trace_log.markdown('')
    st.stop()

# we have data, but no question yet
if not question:
    result_container.markdown('Enter your question and click Enter')
    st.stop()

# we have all - can run it
logger = Logger(verbose=True)
try:
    smart_df = SmartDatalake(st.session_state.data, config={
                    "llm": llm, 
                    "conversational": cb_conversational,
                    "enable_cache": True,
                    "save_charts_path": OUTPUT_GPAPH_FOLDER,
                    "save_charts": True,                   
                    "response_parser": StreamlitResponse 
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

    if isinstance(result, (SmartDataframe, pd.DataFrame, pd.Series)):
        result_container.dataframe(result, use_container_width= True, hide_index= True)
    elif isinstance(result, str):
        if OUTPUT_GPAPH_FOLDER in result:
            if os.path.exists(result):
                image = Image.open(result)
                result_container.image(image)
                os.remove(result)
        else:
            result_container.markdown(result, unsafe_allow_html=True)
    else:
        result_container.markdown(result, unsafe_allow_html=True)

except Exception as error: # pylint: disable=W0718,W0702
    error_log.markdown(f'Error: {error}. Track: {traceback.format_exc()}')
    result_container.markdown("Sorry, I can't process your question. It's still POC. I hope this problem will be fixed soon.")

if logger.logs:
    trace_log.text('\n'.join([log_item['msg'] for log_item in logger.logs]))
else:
    trace_log.markdown('No logs yet')


import streamlit as st
import streamlit.components.v1 as components

# https://discuss.streamlit.io/t/prevent-st-text-input-from-triggering-callback-when-losing-focus/37103/3

def streamlit_hack_disable_textarea_submit():
    components.html(
            """
        <script>
        const doc = window.parent.document;

        const textareas = doc.querySelectorAll('textarea');
        textareas.forEach(textarea => {
            textarea.addEventListener('focusout', function(event) {
                event.stopPropagation();
                event.preventDefault();
            });
        });

        const inputs = doc.querySelectorAll('input');
        inputs.forEach(input => {
            input.addEventListener('focusout', function(event) {
                event.stopPropagation();
                event.preventDefault();
            });
        });
        
        </script>""",
            height=0,
            width=0,
        )

def streamlit_hack_remove_top_space():
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
               .css-1544g2n  {
                    padding-top: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

def streanlit_hide_main_menu():
    st.markdown(
        """
        <style>
        .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
        .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
        .viewerBadge_text__1JaDK, .stToolbar {
            display: none;
        }
        </style>
        """,
    unsafe_allow_html=True
    )
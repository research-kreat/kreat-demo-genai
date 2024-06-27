# main_app.py

import streamlit as st
from spark_blocks import spark_blocks_app
from build_blocks import build_blocks_app

def main():
    st.set_page_config(page_title="Kreat Demo",page_icon="💡")
    pages = {
        "Spark Blocks": spark_blocks_app,
        "Build Blocks": build_blocks_app,
    }

    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Execute the selected page function
    pages[page_selection]()

if __name__ == "__main__":
    main()

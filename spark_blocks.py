# spark_blocks.py

import streamlit as st
from langchain_openai import AzureChatOpenAI

# Function to initialize AzureChatOpenAI
def initialize_llm():
    api_key = st.secrets["azure"]["AZURE_OPENAI_API_KEY"]
    api_version = st.secrets["azure"]["AZURE_OPENAI_API_VERSION"]
    deployment_name = st.secrets["azure"]["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
    endpoint = st.secrets["azure"]["AZURE_OPENAI_ENDPOINT"]

    llm = AzureChatOpenAI(
        openai_api_key=api_key,
        openai_api_version=api_version,
        azure_deployment=deployment_name,
        azure_endpoint=endpoint
    )
    return llm

# Function to classify problem statement for "Spark Blocks"
def classify(llm, problem):
    prompt = f'''
    You are Kreat.ai who helps innovators with their innovations. 
    Given a problem statement. Your job is classify the problem statement into one of the following:
    1. Problem  
    2. Possibility
    3. Idea
    4. Moonshot(IFR)

    ## Definitions and boundaries of each type ##

    ### POSSIBILITIES ###
    Definition: Potential options or alternatives that could be explored or realized, often broader and more abstract than specific ideas or opportunities.
    When to use:
    Exploring potential futures without commitment
    Brainstorming without constraints
    Considering high-level strategic directions
    Engaging in foresight exercises
    Example: "The possibility of brain-computer interfaces becoming mainstream in the next 20 years."
    Key characteristics:
    Broad and abstract
    Future-oriented
    Speculative
    Can lead to multiple ideas or opportunities

    ### IDEAS ### 
    Definition: Specific thoughts or suggestions for a product, service, or solution.
    When to use:
    Proposing concrete solutions or innovations
    Responding to identified problems or needs
    Suggesting improvements to existing products/services
    Example: "A smart refrigerator that automatically orders groceries when supplies are low."
    Key characteristics:
    Specific and concrete
    Actionable
    Often combines existing concepts in novel ways
    Can be developed and implemented


    ### PROBLEMS ###
    Definition: Situations or challenges that need to be addressed, solved, or resolved.
    When to use:
    Identifying barriers or obstacles to progress
    Describing negative situations that require solutions
    Highlighting gaps between current and desired states
    Example: "High employee turnover in our customer service department."
    Key characteristics:
    Describes a negative or undesirable situation
    Often measurable or observable
    Implies a need for a solution
    Can lead to the identification of needs or generation of ideas


    ### MOONSHOTS / IDEAL FINAL RESULTS (IFR) ###
    Definition: Highly ambitious, seemingly impossible goals (Moonshots) or perfect solutions without compromises (IFR).
    When to use:
    Setting audacious, transformative goals
    Imagining perfect solutions without constraints
    Pushing the boundaries of what's considered possible
    Inspiring radical innovation
    Example (Moonshot): "Reverse aging and extend human lifespan to 150 years." Example (IFR): "A transportation system that is instantaneous, free, and has zero environmental impact."
    Key characteristics:
    Extremely ambitious or idealistic
    Often seems impossible with current technology
    Inspires breakthrough thinking
    Challenges fundamental assumptions

    ## Guidelines for you to classify ##
    If your thought is about a potential future development without specific implementation details, it's likely a POSSIBILITY.
    If you have a specific, actionable proposal for a product, service, or solution, it's probably an IDEA.
    If you're describing a negative situation or challenge that needs addressing, you're likely identifying a PROBLEM.
    If your concept is extremely ambitious, seemingly impossible, or describes a perfect solution without compromises, it fits into the MOONSHOT/IFR category.


    Now classify this Problem Statement into one of the types. Think step by Step. 
    Explain your reasoning stepwise as to why it is classified in that category and why is it not classified in the other categories.
    Strictly use the above guidelines and definations of the classification given above.
    Give the output in markdown format.
    Problem: {problem}
    Answer:
    '''
    response = llm.invoke(prompt)
    return response.content

def spark_blocks_app():
    st.title("Kreat.ai Spark Blocks")
    st.header("Classify Your Problem Statement")
    
    llm = initialize_llm()  # Initialize AzureChatOpenAI
    
    problem = st.text_input("Enter your problem statement here:")
    
    if st.button("Classify"):
        if problem:
            classification = classify(llm, problem)
            st.markdown(classification)
        else:
            st.error("Please enter a problem statement.")

if __name__ == "__main__":
    spark_blocks_app()

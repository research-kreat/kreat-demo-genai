# build_blocks.py

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

# Function to classify problem statement for "Build Blocks"
def classify_build_blocks(llm, problem):
    prompt = f"""
        To determine with certainty whether a statement represents a Need, Opportunity, or Concept, we need to look for specific characteristics and linguistic patterns. Here's a guide for each:

        ## Definitions and Boundaries of Each Type ##

        ### NEED ###
        **Definition:** Expresses a requirement or necessity, often implying a gap between the current and desired state.
        **When to use:**
        - Identifying what's lacking or required
        - Highlighting a gap between current and desired conditions
        - Stating a clear requirement

        **Key characteristics:**
        - Focuses on what is lacking or required
        - Often uses words like "need," "require," "must have," "essential"
        - Can be phrased as a lack of something: "lack of," "absence of," "insufficient"
        - May be phrased as a comparative: "more," "better," "faster"

        **Example:** "Employees need more efficient communication tools to collaborate effectively."

        **Certainty indicators:**
        - Clear expression of a requirement
        - Identifiable stakeholder or group with the need
        - Measurable or observable gap

        ### OPPORTUNITY ###
        **Definition:** Represents a favorable circumstance for action or improvement, often tied to market conditions, trends, or organizational capabilities.
        **When to use:**
        - Identifying potential for gain or advantage
        - Highlighting favorable market conditions or trends
        - Suggesting potential for action or improvement

        **Key characteristics:**
        - Represents a favorable circumstance for action or improvement
        - Often uses positive, future-oriented language
        - Often includes words like "potential," "chance," "prospect," "possibility for"
        - May reference market trends or emerging technologies

        **Example:** "The growing demand for sustainable packaging presents an opportunity for eco-friendly product innovation."

        **Certainty indicators:**
        - Clear link to external trends or internal capabilities
        - Identifiable potential for value creation
        - Actionable within the current or near-future context

        ### CONCEPT ###
        **Definition:** Represents a defined idea or proposed solution, more developed than a basic idea but not yet a full plan.
        **When to use:**
        - Proposing a defined idea or solution
        - Combining multiple elements or approaches into a proposed solution
        - Outlining a detailed idea or method

        **Key characteristics:**
        - Represents a defined idea or proposed solution
        - Typically uses descriptive language to outline a solution
        - May use phrases like "a system for," "an approach to," "a method of"
        - Often includes multiple components or steps

        **Example:** "A blockchain-based supply chain tracking system that ensures product authenticity and ethical sourcing."

        **Certainty indicators:**
        - Clearly defined proposed solution or approach
        - Combination of multiple elements or ideas
        - More detailed than a basic idea, but not as specific as a full plan

        To determine with certainty, consider these additional factors:
        1. Context: The surrounding information or discussion can provide clues about whether something is being presented as a need, opportunity, or concept.
        2. Source: The origin of the statement (e.g., market research, customer feedback, R&D team) can indicate its classification.
        3. Intended use: How the statement is meant to be used (e.g., to guide product development, to inform strategy) can help classify it.
        4. Level of detail: Needs are often broader, opportunities are tied to specific circumstances, and concepts have more detailed descriptions.
        5. Action orientation: Needs imply something to be fulfilled, opportunities suggest something to be seized, and concepts propose something to be developed.

        By carefully analyzing these aspects, you can categorize statements with a high degree of certainty. However, remember that in some cases, a statement might have characteristics of multiple categories, in which case you'd need to determine the primary intent or most prominent features to make a final classification.

        Now classify this Problem Statement into one of the types. Think step by step. Explain your reasoning stepwise as to why it is classified in that category and why it is not classified in the other categories. Strictly use the above guidelines and definitions of the classification given above.

        Problem: {problem}
        Answer:
        """
    response = llm.invoke(prompt)
    return response.content

def build_blocks_app():
    st.title("Kreat.ai Build Blocks")
    st.header("Classify Your Problem into Appropriate Building Blocks")
    
    llm = initialize_llm()  # Initialize AzureChatOpenAI
    
    problem = st.text_input("Enter your problem statement here:")
    
    if st.button("Classify"):
        if problem:
            classification = classify_build_blocks(llm, problem)
            st.markdown(classification)
        else:
            st.error("Please enter a problem statement.")

if __name__ == "__main__":
    build_blocks_app()

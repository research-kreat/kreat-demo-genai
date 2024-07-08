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


# Function to prepare a problem title
def generate_title(llm, extracted_problem):
    prompt = f'''
        You are tasked with generating Title for a problem statement of an innovator from some information provided to you by the innovator.
        Here are some information related to a problem. 
        {extracted_problem} 
        Using these extracted elements, create a problem statement that follows these guidelines:

        1. Scope indication:
        Characteristic: Includes a hint about the scale or scope of the problem.
        Good example: "Reducing Plastic Waste in Southeast Asian Coastal Communities"
        Bad example: "Plastic Waste Reduction"

        2. Stakeholder focus:
        Characteristic: Mentions key stakeholders affected by or involved in the problem.
        Good example: "Improving Healthcare Access for Rural Elderly Populations"
        Bad example: "Healthcare Access Improvement"

        3. Timeframe:
        Characteristic: Indicates whether it's an urgent, ongoing, or future issue.
        Good example: "Addressing Immediate Food Insecurity in Drought-Affected Regions"
        Bad example: "Food Insecurity in Drought Regions"

        4. Outcome-oriented:
        Characteristic: Suggests the desired result or improvement.
        Good example: "Enhancing Student Engagement Through Gamified Learning Platforms"
        Bad example: "Using Gamification in Education"

        5. Keyword optimization:
        Characteristic: Uses relevant keywords for searchability and categorization.
        Good example: "Sustainable Urban Development: Implementing Green Infrastructure Solutions"
        Bad example: "City Planning Improvements"

        6. Avoid unnecessary words:
        Characteristic: Eliminates articles and filler words when possible.
        Good example: "Reducing Industrial Carbon Emissions"
        Bad example: "The Challenge of Reducing the Carbon Emissions in the Industry"

        7. Use active voice:
        Characteristic: Employs active rather than passive language for directness.
        Good example: "Implementing Water Conservation Strategies in Arid Regions"
        Bad example: "Water Conservation Strategies Being Implemented in Arid Regions"

        8. Quantify if possible:
        Characteristic: Includes numbers or metrics if they add significant value.
        Good example: "Halving Food Waste: A 10-Year Strategy for Restaurants"
        Bad example: "Reducing Food Waste in Restaurants"

        9. Avoid questions:
        Characteristic: Frames the title as a statement rather than a question.
        Good example: "Improving Public Transportation Efficiency in High-Density Urban Areas"
        Bad example: "How Can We Improve Public Transportation in Crowded Cities?"

        10. Balance creativity and clarity:
            Characteristic: Uses engaging language but prioritizes clarity over cleverness.
            Good example: "From Trash to Treasure: Upcycling Industrial Waste into Valuable Products"
            Bad example: "Turning Garbage into Gold: A Waste Revolution"

        11. Consistency:
            Characteristic: Ensures the title aligns with the content of the problem statement.
            Good example: "Global Climate Change Mitigation Strategies" (assuming the content discusses global strategies)
            Bad example: "Global Climate Change Mitigation Strategies" (if the content only discusses local initiatives)

        12. Avoid abbreviations:
            Characteristic: Spells out terms unless universally recognized in the field.
            Good example: "Reducing Greenhouse Gas Emissions in the Transport Sector"
            Bad example: "Reducing GHG Emissions in the Transport Sector"

        13. Clarity and Simplicity:
            Characteristic: Ensure the title is easy to understand and free of complex jargon unless necessary.
            Good example: "Improving Air Quality in Urban Areas"
            Bad example: "Enhancing Atmospheric Composition Through Pollution Mitigation"

        14. Engagement:
            Characteristic: Make the title engaging to capture the reader's interest.
            Good example: "Boosting Renewable Energy Adoption in Developing Countries"
            Bad example: "Promoting Renewable Energy"

        15. Precision:
            Characteristic: Use precise and specific language to avoid vagueness.
            Good example: "Enhancing Cybersecurity Measures in Online Banking"
            Bad example: "Improving Security in Banking"

        16. Length:
            Characteristic: Maintain a balance between brevity and informativeness, aiming for 5 to 12 words.
            Good example: "Enhancing Urban Mobility Through Bike-Sharing Programs"
            Bad example: "Exploring Ways to Enhance Urban Mobility Through the Implementation of Bike-Sharing Programs"

        17. Perspective:
            Characteristic: Reflect the perspective or approach being taken, such as policy, technology, or societal impact.
            Good example: "Policy Interventions for Reducing Childhood Obesity"
            Bad example: "Childhood Obesity Reduction"
        '''
    response = llm.invoke(prompt)
    return response.content


#Funtion to check a problem title
def check_title(llm,title):
    guidelines = [
        "1. Scope indication: Includes a hint about the scale or scope of the problem. Good example: 'Reducing Plastic Waste in Southeast Asian Coastal Communities'. Bad example: 'Plastic Waste Reduction'.",
        "2. Stakeholder focus: Mentions key stakeholders affected by or involved in the problem. Good example: 'Improving Healthcare Access for Rural Elderly Populations'. Bad example: 'Healthcare Access Improvement'.",
        "3. Timeframe: Indicates whether it's an urgent, ongoing, or future issue. Good example: 'Addressing Immediate Food Insecurity in Drought-Affected Regions'. Bad example: 'Food Insecurity in Drought Regions'.",
        "4. Outcome-oriented: Suggests the desired result or improvement. Good example: 'Enhancing Student Engagement Through Gamified Learning Platforms'. Bad example: 'Using Gamification in Education'.",
        "5. Keyword optimization: Uses relevant keywords for searchability and categorization. Good example: 'Sustainable Urban Development: Implementing Green Infrastructure Solutions'. Bad example: 'City Planning Improvements'.",
        "6. Avoid unnecessary words: Eliminates articles and filler words when possible. Good example: 'Reducing Industrial Carbon Emissions'. Bad example: 'The Challenge of Reducing the Carbon Emissions in the Industry'.",
        "7. Use active voice: Employs active rather than passive language for directness. Good example: 'Implementing Water Conservation Strategies in Arid Regions'. Bad example: 'Water Conservation Strategies Being Implemented in Arid Regions'.",
        "8. Quantify if possible: Includes numbers or metrics if they add significant value. Good example: 'Halving Food Waste: A 10-Year Strategy for Restaurants'. Bad example: 'Reducing Food Waste in Restaurants'.",
        "9. Avoid questions: Frames the title as a statement rather than a question. Good example: 'Improving Public Transportation Efficiency in High-Density Urban Areas'. Bad example: 'How Can We Improve Public Transportation in Crowded Cities?'.",
        "10. Balance creativity and clarity: Uses engaging language but prioritizes clarity over cleverness. Good example: 'From Trash to Treasure: Upcycling Industrial Waste into Valuable Products'. Bad example: 'Turning Garbage into Gold: A Waste Revolution'.",
        "11. Consistency: Ensures the title aligns with the content of the problem statement. Good example: 'Global Climate Change Mitigation Strategies'. Bad example: 'Global Climate Change Mitigation Strategies' (if the content only discusses local initiatives).",
        "12. Avoid abbreviations: Spells out terms unless universally recognized in the field. Good example: 'Reducing Greenhouse Gas Emissions in the Transport Sector'. Bad example: 'Reducing GHG Emissions in the Transport Sector'.",
        "13. Clarity and Simplicity: Ensure the title is easy to understand and free of complex jargon unless necessary. Good example: 'Improving Air Quality in Urban Areas'. Bad example: 'Enhancing Atmospheric Composition Through Pollution Mitigation'.",
        "14. Engagement: Make the title engaging to capture the reader's interest. Good example: 'Boosting Renewable Energy Adoption in Developing Countries'. Bad example: 'Promoting Renewable Energy'.",
        "15. Precision: Use precise and specific language to avoid vagueness. Good example: 'Enhancing Cybersecurity Measures in Online Banking'. Bad example: 'Improving Security in Banking'.",
        "16. Length: Maintain a balance between brevity and informativeness, aiming for 5 to 12 words. Good example: 'Enhancing Urban Mobility Through Bike-Sharing Programs'. Bad example: 'Exploring Ways to Enhance Urban Mobility Through the Implementation of Bike-Sharing Programs'.",
        "17. Perspective: Reflect the perspective or approach being taken, such as policy, technology, or societal impact. Good example: 'Policy Interventions for Reducing Childhood Obesity'. Bad example: 'Childhood Obesity Reduction'."
    ]

    prompt = f'''
    Given title: "{title}"
    Please evaluate this title based on the following guidelines:
    {guidelines}
    Provide a clear judgment: "Yes it's a good title" or "No if it is out of the scope". If it's no then give clear reasons as to which guidelines it failed.
    '''
    response = llm.invoke(prompt)
    return response.content

# Function to extract problems from a conversation
def problem_extraction(llm, problem):
    prompt = f''' 
            {problem}

            Based on the innovator's response to "What is a problem for you?", extract the following key elements:

            1. Core issue: Identify the central problem or challenge.
            2. Affected stakeholders: Note who is impacted by this problem.
            3. Context or scope: Determine the scale and relevant setting of the problem.
            4. Current impact: Understand how this problem is affecting stakeholders or the situation.
            5. Desired outcome: If mentioned, note the envisioned improvement or solution.
            6. Root causes: Identify underlying factors contributing to the problem.
            7. Timeframe: Note if the problem is urgent, ongoing, or anticipated.
            8. Quantifiable aspects: Extract any numbers, statistics, or metrics mentioned.
            9. Industry or field: Determine the specific sector or area of focus.
            10. Key terms: Identify important keywords or phrases related to the problem.
            11. Constraints: Note any limitations or obstacles in addressing the problem.
            12. Unique aspects: Highlight distinctive features of the problem.

            '''
    response = llm.invoke(prompt)
    return response.content

def update_abstract(llm, abstract, feedback):
    prompt = f'''
        You are tasked with understand the sentiment of a feedback and updating an abstract if required based on the feedback provided by the user.
        Here is the original abstract:
        {abstract}
        
        And here is the feedback:
        {feedback}
        
        If the feedback is positive then your response will the same abstract otherwise please update the abstract according to the feedback. 
        Thus your response will be one of the following:

        Output:

        Okay then we will stick with the same abstract:
        {abstract}

        or

        Okay here's an updated abstract:
        
    '''
    
    response = llm.invoke(prompt)
    return response.content

def update_title(llm, title, feedback):
    prompt = f'''
        You are tasked with understand the sentiment of a feedback and updating an abstract if required based on the feedback provided by the user.
        Here is the original Title:
        {title}
        
        And here is the feedback:
        {feedback}
        
        If the feedback is positive then your response will the same abstract otherwise please update the abstract according to the feedback. 
        Thus your response will be one of the following:

        Output:

        Okay then we will stick with the same Title:
        {title}

        or

        Okay here's an updated Title:
        
    '''
    
    response = llm.invoke(prompt)
    return response.content



# Function to prepare an abstract
def generate_abstract(llm, title,extracted_problem):
    prompt = f'''
        You are tasked with generating an abstract for a problem statement from some information provided to you by the innovator.
        Here is the title of the problem:
        {title}
        Here are some information related to a problem. 
        {extracted_problem} 
        Using these extracted elements, create an abstract that follows these guidelines:

        1. Conciseness:
        Characteristic: Uses as few words as possible to convey the core message.

        2. Clarity:
        Characteristic: Clearly states the problem, avoiding ambiguous terms and jargon.

        3. Relevance:
        Characteristic: Highlights why the problem matters and to whom.

        4. Specificity:
        Characteristic: Provides specific details about the problem.

        5. Impact:
        Characteristic: Describes the potential consequences of the problem.

        6. Scope:
        Characteristic: Indicates the scope or scale of the issue.

        7. Solution Hint:
        Characteristic: Hints at potential solution areas or interventions.

        8. Quantifiable Goals:
        Characteristic: Mentions specific, measurable objectives if possible.

        9. Problem Context:
        Characteristic: Provides a brief context or background to the problem.

        10. Stakeholder Impact:
        Characteristic: Identifies who is affected by the problem.

        11. Methodology Hint:
        Characteristic: Briefly hints at the approach or methodology to address the problem.

        12. Timeframe:
        Characteristic: Indicates whether it's an urgent, ongoing, or future issue.

        13. Outcome Expectation:
        Characteristic: Suggests the potential positive outcomes.

        14. Engagement:
        Characteristic: Engages the reader’s interest.

        15. Precision:
        Characteristic: Uses precise and specific language to avoid vagueness.

        Here are some examples:

        Example 1:
        "Urban air pollution, primarily caused by vehicle emissions, affects the health of millions in major cities. This project aims to reduce pollution levels by 30% through the implementation of AI-driven traffic management systems. This intervention could significantly improve public health outcomes and enhance the quality of life for urban residents. Immediate action is needed to address this pressing issue and to secure a healthier future."

        Example 2:
        "Deforestation in the Amazon rainforest results in significant biodiversity loss and disrupts the lives of indigenous communities. This research focuses on developing sustainable land management practices to preserve this vital ecosystem. By integrating community-driven conservation efforts, we aim to reduce deforestation rates by 40% over the next decade. Successful implementation will help maintain biodiversity and support the livelihoods of local populations."

        Example 3:
        "Water scarcity in arid regions poses a severe threat to both human populations and agricultural activities. Our initiative proposes the adoption of advanced water-saving technologies and practices to increase water use efficiency. With a target of improving water availability by 25% within five years, this project seeks to enhance sustainability and resilience in these vulnerable areas. Timely intervention is crucial to prevent further degradation of water resources."

        Your response should be in the following format:
        Here's a draft abstract for you:
        Abstract:
    '''
    response = llm.invoke(prompt)
    return response.content

# Function to prepare assumptions
def generate_assumptions(llm, extracted_problem):
    prompt = f'''
        You are tasked with generating assumptions for a problem statement from some information provided to you by the innovator.
        Here are some information related to a problem. 
        {extracted_problem} 
        Using these extracted elements, create assumptions that follow these guidelines:

        1. Clearly stated and justified:
        Characteristic: Each assumption should be explicitly stated and accompanied by a rationale.

        2. Relevant to the problem and potential solutions:
        Characteristic: Assumptions should directly relate to the problem at hand and the proposed solutions.

        3. Acknowledges uncertainties:
        Characteristic: Assumptions should recognize any uncertainties or potential variability.

        4. Includes both technical and social assumptions:
        Characteristic: Assumptions should cover both technical aspects and social dynamics.

        Here are some examples:

        Example 1:
        Assumption: "We assume that the majority of vehicles in the city will be equipped with or can be retrofitted with IoT devices for traffic management systems."
        Rationale: This is based on the increasing prevalence of smart vehicles and the city's plan to subsidize IoT device installation.

        Assumption: "We assume that reducing traffic congestion will lead to a proportional reduction in air pollution levels."
        Rationale: This is based on studies showing a strong correlation between traffic density and air pollution in urban areas.

        Example 2:
        Assumption: "We assume that local farmers will adopt the proposed sustainable land management practices."
        Rationale: This is based on prior successful pilot programs and the availability of government incentives.

        Assumption: "We assume that international funding for conservation efforts will remain stable or increase."
        Rationale: This is based on recent trends in global environmental funding and commitments made at international climate summits.

        Example 3:
        Assumption: "We assume that advancements in water-saving technologies will be accessible and affordable to farmers in arid regions."
        Rationale: This is based on current trends in technology development and subsidies offered by the government.

        Assumption: "We assume that public awareness campaigns will effectively change water usage behaviors."
        Rationale: This is based on the success of similar campaigns in other regions and the involvement of local community leaders.

    '''
    response = llm.invoke(prompt)
    return response.content

# Function to prepare a problem description
def generate_description(llm, extracted_problem):
    prompt = f'''
        You are tasked with generating a description for a problem statement from some information provided to you by the innovator.
        Here are some information related to a problem. 
        {extracted_problem} 
        Using these extracted elements, create a description that follows these guidelines:

        1. Detailed context and background:
        Characteristic: Provides detailed context and background about the problem.

        2. Quantifies the problem with data when possible:
        Characteristic: Includes specific numbers or metrics to quantify the problem.

        3. Root causes and contributing factors:
        Characteristic: Explains the root causes and contributing factors of the problem.

        4. Current attempts to solve the problem:
        Characteristic: Describes any current solutions or efforts being made to solve the problem.

        5. Potential impacts of solving (or not solving) the problem:
        Characteristic: Outlines the potential impacts of solving or not solving the problem.

        6. Logical flow of information:
        Characteristic: Ensures the information is presented in a logical, coherent manner.

        7. Contextual relevance:
        Characteristic: Connects the problem to broader social, economic, or environmental issues.

        8. Historical perspective:
        Characteristic: Provides a brief history of how the problem has evolved over time.

        9. Stakeholder analysis:
        Characteristic: Identifies key stakeholders affected by or involved in the problem.

        10. Finally the description should contain step by step breakdown of the problem and not just be a bunch of paragraphs. 

        Here is an example:

        "Climate change poses a significant threat to coastal cities worldwide, with rising sea levels and increased storm intensity. In [City Z], sea levels have risen by an average of 3 millimeters per year over the past two decades, resulting in frequent flooding and coastal erosion. The primary causes of this problem include global warming, melting polar ice caps, and increased greenhouse gas emissions from industrial activities. 

        Current mitigation efforts in [City Z] include the construction of sea walls, the development of early warning systems, and the implementation of green infrastructure projects. However, these measures have been insufficient to fully address the issue. The city's sea walls, for example, are often overwhelmed during severe storms, leading to significant property damage and displacement of residents. 

        If left unaddressed, the impacts of rising sea levels in [City Z] could be catastrophic. By 2050, it is projected that over 100,000 residents could be displaced due to flooding, resulting in economic losses of up to $1 billion annually. Additionally, the loss of coastal habitats could lead to a decline in biodiversity, affecting local fisheries and tourism industries. 

        To effectively combat this issue, a multifaceted approach is required. This includes reducing greenhouse gas emissions, enhancing coastal resilience through natural solutions like mangrove restoration, and improving urban planning to accommodate future sea level rise. Addressing climate change in [City Z] is not only critical for protecting the local population and economy but also contributes to global efforts in mitigating environmental degradation."

    '''
    response = llm.invoke(prompt)
    return response.content

# Function to prepare constraints
def generate_constraints(llm, extracted_problem):
    prompt = f'''
        You are tasked with generating constraints for a problem statement from some information provided to you by the innovator.
        Here are some information related to a problem. 
        {extracted_problem} 
        Using these extracted elements, create constraints that follow these guidelines:

        1. Identifies limiting factors clearly:
        Characteristic: Each constraint should clearly state what limits or restricts the solution.

        2. Includes both technical and non-technical constraints:
        Characteristic: Constraints should cover both technological requirements and broader constraints.

        3. Quantifies limitations where possible:
        Characteristic: Constraints should include specific numbers, figures, or durations where applicable.

        4. Considers regulatory, ethical, and social constraints:
        Characteristic: Constraints should address legal, ethical, and societal considerations.

        Here are some examples:

        Budget Constraint: "The project must be implemented within a budget of $10 million over 3 years."
        Technical Constraint: "The solution must be compatible with the city's existing traffic light infrastructure."
        Regulatory Constraint: "All data collection and use must comply with GDPR and local privacy laws."
        Social Constraint: "The solution must not disproportionately affect low-income areas or exacerbate existing inequalities in transportation access."
        Time Constraint: "The initial phase of the project must show measurable results within 12 months to ensure continued funding."

        Additional Constraints:
        - Environmental Constraint: "The solution must minimize carbon emissions during implementation."
        - Safety Constraint: "The technology must meet industry safety standards to prevent accidents."
        - Scalability Constraint: "The solution should be scalable to accommodate future population growth."

        Please generate specific constraints based on the problem statement provided.
    '''
    response = llm.invoke(prompt)
    return response.content

# Function to prepare risks
def generate_risks(llm, extracted_problem):
    prompt = f'''
        You are tasked with generating risks for a problem statement from some information provided to you by the innovator.
        Here are some information related to a problem. 
        {extracted_problem} 
        Using these extracted elements, create risks that follow these guidelines:

        1. Clearly identified and categorized:
        Characteristic: Each risk should be clearly identified and categorized (e.g., technical, financial, social).

        2. Assessed for both likelihood and potential impact:
        Characteristic: Risks should be assessed for their likelihood of occurring and their potential impact if they do occur.

        3. Includes potential mitigation strategies:
        Characteristic: Each risk should include potential strategies to mitigate or reduce its likelihood or impact.

        4. Considers short-term and long-term risks:
        Characteristic: Risks should be evaluated for both short-term and long-term implications.

        Here are some examples:

        Technical Risk: "AI algorithms may initially misinterpret traffic patterns, leading to suboptimal traffic management."
        Mitigation: Extensive testing in simulated environments and gradual rollout with human oversight.

        Financial Risk: "Unexpected budget overruns due to unforeseen costs in technology implementation."
        Mitigation: Regular financial audits and contingency planning for cost escalations.

        Social Risk: "Public resistance to perceived 'surveillance' through traffic monitoring systems."
        Mitigation: Transparent communication campaign and strict data anonymization protocols.

        Legal Risk: "Potential lawsuits or fines due to non-compliance with new data privacy regulations."
        Mitigation: Legal review of all data handling practices and continuous monitoring of regulatory changes.

        Operational Risk: "System downtime during peak traffic hours, impacting service reliability."
        Mitigation: Redundancy planning and regular maintenance schedules to minimize downtime.

        Please generate specific risks based on the problem statement provided.
    '''
    response = llm.invoke(prompt)
    return response.content


def convo():
    st.title("Kreat Conversation")
    st.header("Converse with Kreat")


     # Sidebar navigation
    st.sidebar.title("Function Navigator")
    function_names = [
        "Generate Title",
        "Update Title",
        "Generate Abstract",
        "Update Abstract",
        "Assess Problem Type",
        "Explain Problem Type Assessment",
        "Visualize Sliders",
        "Access Data Sources",
        "Summarize Key Findings",
        "Update Problem Description",
        "Analyze Problem Breadth and Depth",
        "Update Breadth and Depth",
        "Generate Future Scenarios",
        "Create Function Map",
        "Apply TRIZ Principle",
        "Generate Problem Summary",
        "Recommend Experts",
        "Create Visual Map",
        "Download Analysis",
        "Share Analysis"
    ]

    choice = st.sidebar.selectbox("Select a Function", function_names)
    llm = initialize_llm()  # Initialize AzureChatOpenAI

    # Main content based on sidebar choice
    st.title(choice)

    if choice == "Generate Title":
        goal = st.text_input("Goal")
        if st.button("Run"):
            with st.spinner("Extracting Information...."):
                extracted_information = problem_extraction(llm,goal)
            result = generate_title(llm,extracted_information)
            st.write(result)

    elif choice == "Update Title":
        current_title = st.text_input("Current Title")
        feedback = st.text_input("Feedback")
        if st.button("Run"):
            result = update_title(llm,current_title, feedback)
            st.write(result)

    elif choice == "Generate Abstract":
        title = st.text_input("Title")
        if st.button("Run"):
            result = generate_abstract(llm,title,extracted_information)
            st.write(result)

    elif choice == "Update Abstract":
        current_abstract = st.text_input("Current Abstract")
        feedback = st.text_input("Feedback")
        if st.button("Run"):
            result = update_abstract(current_abstract, feedback)
            st.write(result)

    elif choice == "Assess Problem Type":
        st.write("Prompt Under Development")
    elif choice == "Explain Problem Type Assessment":
        st.write("Prompt Under Development")

    elif choice == "Visualize Sliders":
        st.write("Prompt Under Development")

    elif choice == "Access Data Sources":
        st.write("Prompt Under Development")
    elif choice == "Summarize Key Findings":
        pass
    elif choice == "Update Problem Description":
        st.write("Prompt Under Development")

    elif choice == "Analyze Problem Breadth and Depth":
        st.write("Prompt Under Development")
    elif choice == "Update Breadth and Depth":
        st.write("Prompt Under Development")

    elif choice == "Generate Future Scenarios":
        st.write("Prompt Under Development")

    elif choice == "Create Function Map":
        st.write("Prompt Under Development")
    elif choice == "Apply TRIZ Principle":
        st.write("Prompt Under Development")

    elif choice == "Generate Problem Summary":
        # title = st.text_input("Title")
        # abstract = st.text_input("Abstract")
        # problem_description = st.text_input("Problem Description")
        # insights = st.text_area("Insights (comma-separated)").split(",")
        # future_scenarios = st.text_area("Future Scenarios (comma-separated)").split(",")
        # function_map = st.text_input("Function Map")
        # triz_solution = st.text_input("TRIZ Solution")
        # if st.button("Run"):
        #     result = generate_problem_summary(title, abstract, problem_description, insights, future_scenarios, function_map, triz_solution)
        #     st.write(result)
        st.write("Prompt Under Development")

    elif choice == "Recommend Experts":
        st.write("Prompt Under Development")
    elif choice == "Create Visual Map":
        st.write("Prompt Under Development")

    elif choice == "Download Analysis":
        st.write("Prompt Under Development")

    elif choice == "Share Analysis":
        st.write("Prompt Under Development")
    
if __name__ == "__main__":
    convo()



#to be inserted after problem extraction

#title = problem_title(llm,extracted_problem)
# st.markdown('### Problem title ###')
# st.markdown(title)

# abstract = generate_abstract(llm,extracted_problem)
# st.markdown('### Problem abstract ###')
# st.markdown(abstract)

# assumptions = generate_assumptions(llm,extracted_problem)
# st.markdown('### Problem assumptions ###')
# st.markdown(assumptions)

# description = generate_description(llm,extracted_problem)
# st.markdown('### Problem description ###')
# st.markdown(description)


# constraints = generate_constraints(llm,extracted_problem)
# st.markdown('### Problem constraints ###')
# st.markdown(constraints)


# risks = generate_risks(llm,extracted_problem)
# st.markdown('### Problem risks ###')
# st.markdown(risks)




# if 'problem' not in st.session_state:
#         st.session_state.problem = ""
#     if 'extracted_problem' not in st.session_state:
#         st.session_state.extracted_problem = ""
#     if 'abstract' not in st.session_state:
#         st.session_state.abstract = ""
#     if 'title' not in st.session_state:
#         st.session_state.title = ""
#     if 'show_options' not in st.session_state:
#         st.session_state.show_options = False
#     if 'abstract_generated' not in st.session_state:
#         st.session_state.abstract_generated = False
#     if 'title_generated' not in st.session_state:
#         st.session_state.title_generated = False
#     if 'description_generated' not in st.session_state:
#         st.session_state.description_generated = False

#     st.session_state.problem = st.text_area("Hey, what is the problem that you are trying to crack?", st.session_state.problem)

#     if st.button("Enter", key="problem-input"):
#         if st.session_state.problem:
#             st.session_state.show_options = True
#             st.session_state.extracted_problem = problem_extraction(llm, st.session_state.problem)
#         else:
#             st.error("Please enter a problem statement.")

#     if st.session_state.show_options:
#         st.markdown("**Cool, should we make a good abstract based on this?**")
#         option = st.radio(
#             "Please choose an option:",
#             ('Yes get me an AI generated abstract', 'No I will provide an abstract')
#         )

#         if option == 'Yes get me an AI generated abstract':
#             if not st.session_state.abstract_generated:
#                 with st.spinner('Generating Abstracts'):
#                     st.session_state.abstract = generate_abstract(llm, st.session_state.extracted_problem)
#                     st.session_state.abstract_generated = True
#             st.markdown('### Abstract ###')
#             st.markdown(st.session_state.abstract)
            
#             abstract_feedback = st.text_area("Do you like the abstract or should we change something? Please let me know below:", key="abstract-feedback")
#             if st.button("Update Abstract", key="update-abstract"):
#                 with st.spinner('Finishing up the abstract...'):
#                     st.session_state.abstract = update_abstract(llm, st.session_state.abstract, abstract_feedback)
#                 st.markdown('### Abstract ###')
#                 st.markdown(st.session_state.abstract)
                
#             if not st.session_state.title_generated and st.button("Generate Title", key="generate-title"):
#                 with st.spinner('Generating Title'):
#                     st.session_state.title = problem_title(llm, st.session_state.extracted_problem)
#                     st.session_state.title_generated = True
#                 st.markdown('### Problem Title ###')
#                 st.markdown(st.session_state.title)

#             if st.session_state.title_generated:
#                 st.markdown('### Problem Title ###')
#                 st.markdown(st.session_state.title)
#                 title_feedback = st.text_area("Do you like the title or should we change something? Please let me know below:", key="title-feedback")
#                 if st.button("Update Title", key="update-title"):
#                     with st.spinner('Finishing up the title...'):
#                         st.session_state.title = update_title(llm, st.session_state.title, title_feedback)
#                     st.markdown('### Problem Title ###')
#                     st.markdown(st.session_state.title)
                    
#                     if not st.session_state.description_generated:
#                         if st.button("Generate Description", key="generate-description"):
#                             with st.spinner('Generating Description'):
#                                 st.session_state.description = generate_description(llm, st.session_state.extracted_problems)
#                                 st.session_state.description_generated = True
#                             st.markdown('### Generated Description ###')
#                             st.markdown(st.session_state.description)
#                     elif st.session_state.description_generated:
#                         st.markdown('### Generated Description ###')
#                         st.markdown(st.session_state.description)

#         else:
#             st.session_state.abstract = st.text_area('You may enter an abstract:')
#             st.session_state.title = st.text_area('You may enter a title:')

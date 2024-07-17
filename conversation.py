# build_blocks.py

import streamlit as st
from langchain_openai import AzureChatOpenAI
import pandas as pd
from exa_py import Exa
import json

#from langchain_groq import ChatGroq

# Function to initialize AzureChatOpenAI
def initialize_llm():
    api_key = st.secrets["azure"]["AZURE_OPENAI_API_KEY"]
    api_version = st.secrets["azure"]["AZURE_OPENAI_API_VERSION"]
    deployment_name = st.secrets["azure"]["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
    endpoint = st.secrets["azure"]["AZURE_OPENAI_ENDPOINT"]

    #llm = ChatGroq(temperature=0.5, groq_api_key="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", model_name="llama3-70b-8192")

    llm = AzureChatOpenAI(
        openai_api_key=api_key,
        openai_api_version=api_version,
        azure_deployment=deployment_name,
        azure_endpoint=endpoint,
        temperature=0.5
    )
    return llm

exa = Exa(st.secrets["exa"]["EXA_API_KEY"])

class Result:
    def __init__(self, url, id, title, score, published_date, author):
        self.url = url
        self.id = id
        self.title = title
        self.score = score
        self.published_date = published_date
        self.author = author

    def to_dict(self):
        return {
            "url": self.url,
            "id": self.id,
            "title": self.title,
            "score": self.score,
            "published_date": self.published_date,
            "author": self.author
        }

class SearchResponse:
    def __init__(self, results, autoprompt_string=None):
        self.results = results
        self.autoprompt_string = autoprompt_string

def extract_search_results(search_response):
    extracted_results = [result.to_dict() for result in search_response.results]
    return extracted_results

# Function to perform the search and extract results
def search_and_extract(query, include_domains=None, start_published_date=None):
    # Perform the search
    search_response = exa.search_and_contents(query,use_autoprompt=True,num_results=3)
    
    return search_response.results


def problem_extraction(llm, problem):
    prompt = f'''
    {problem}

    Based on the innovator's response to "What is a problem for you?", extract the following key elements and present them in the specified format:

    CORE ISSUE: "....[Identify the central problem or challenge]..."

    AFFECTED STAKEHOLDERS: "....[Note who is impacted by this problem]..."

    CONTEXT OR SCOPE: "....[Determine the scale and relevant setting of the problem]..."

    CURRENT IMPACT: "....[Understand how this problem is affecting stakeholders or the situation]..."

    DESIRED OUTCOME: "....[If mentioned, note the envisioned improvement or solution]..."

    ROOT CAUSES: "....[Identify underlying factors contributing to the problem]..."

    TIMEFRAME: "....[Note if the problem is urgent, ongoing, or anticipated]..."

    QUANTIFIABLE ASPECTS: "....[Extract any numbers, statistics, or metrics mentioned]..."

    INDUSTRY OR FIELD: "....[Determine the specific sector or area of focus]..."

    KEY TERMS: "....[Identify important keywords or phrases related to the problem]..."

    CONSTRAINTS: "....[Note any limitations or obstacles in addressing the problem]..."

    UNIQUE ASPECTS: "....[Highlight distinctive features of the problem]..."

    Please provide your response in the exact format shown above, with each category in capital letters followed by a colon and the answers should be within double quotation"... ...". If information for a category is not available or not mentioned, write "Not specified" for that category.
    '''
    response = llm.invoke(prompt)
    return response.content

def parse_problem_extraction(output):
    # Initialize a dictionary to store the parsed values
    parsed_data = {
        "CORE ISSUE": "",
        "AFFECTED STAKEHOLDERS": "",
        "CONTEXT OR SCOPE": "",
        "CURRENT IMPACT": "",
        "DESIRED OUTCOME": "",
        "ROOT CAUSES": "",
        "TIMEFRAME": "",
        "QUANTIFIABLE ASPECTS": "",
        "INDUSTRY OR FIELD": "",
        "KEY TERMS": "",
        "CONSTRAINTS": "",
        "UNIQUE ASPECTS": ""
    }

    # Split the output by lines
    lines = output.split('\n')

    current_key = None

    for line in lines:
        line = line.strip()
        if line:
            # Check if the line starts with any of our keys
            for key in parsed_data.keys():
                if line.startswith(key + ":"):
                    current_key = key
                    parsed_data[key] = line.split(":", 1)[1].strip()
                    break
            else:
                # If no key is found, it's a continuation of the previous value
                if current_key:
                    parsed_data[current_key] += " " + line

    return parsed_data

def generate_title(llm, extracted_problem):
    prompt = f'''
    You are tasked with generating a Title for a problem statement of an innovator from some information provided to you by the innovator.
    Here is some information related to a problem:
    {extracted_problem}
    Using these extracted elements, create a problem statement title that follows the guidelines provided below.

    Guidelines for creating the title:
    1. Scope indication: Include a hint about the scale or scope of the problem.
    2. Stakeholder focus: Mention key stakeholders affected by or involved in the problem.
    3. Timeframe: Indicate whether it's an urgent, ongoing, or future issue.
    4. Outcome-oriented: Suggest the desired result or improvement.
    5. Keyword optimization: Use relevant keywords for searchability and categorization.
    6. Avoid unnecessary words: Eliminate articles and filler words when possible.
    7. Use active voice: Employ active rather than passive language for directness.
    8. Quantify if possible: Include numbers or metrics if they add significant value.
    9. Avoid questions: Frame the title as a statement rather than a question.
    10. Balance creativity and clarity: Use engaging language but prioritize clarity over cleverness.
    11. Consistency: Ensure the title aligns with the content of the problem statement.
    12. Avoid abbreviations: Spell out terms unless universally recognized in the field.
    13. Clarity and Simplicity: Ensure the title is easy to understand and free of complex jargon unless necessary.
    14. Engagement: Make the title engaging to capture the reader's interest.
    15. Precision: Use precise and specific language to avoid vagueness.
    16. Length: Maintain a balance between brevity and informativeness, aiming for 5 to 12 words.
    17. Perspective: Reflect the perspective or approach being taken, such as policy, technology, or societal impact.


    **Output Format:**

    TITLE: "......"

    EVALUATION:
    SCOPE INDICATION: "YES/NO"
    STAKEHOLDER FOCUS: "YES/NO"
    TIMEFRAME: "YES/NO"
    OUTCOME-ORIENTED: "YES/NO"
    KEYWORD OPTIMIZATION: "YES/NO"
    AVOID UNNECESSARY WORDS: "YES/NO"
    USE ACTIVE VOICE: "YES/NO"
    QUANTIFY IF POSSIBLE: "YES/NO"
    AVOID QUESTIONS: "YES/NO"
    BALANCE CREATIVITY AND CLARITY: "YES/NO"
    CONSISTENCY: "YES/NO"
    AVOID ABBREVIATIONS: "YES/NO"
    CLARITY AND SIMPLICITY: "YES/NO"
    ENGAGEMENT: "YES/NO"
    PRECISION: "YES/NO"
    LENGTH: "YES/NO"
    PERSPECTIVE: "YES/NO"

    
    Note: Please follow the exact output format while answering. The Title response should be within double quotation marks. Each YES/NO evaluation should be within double quotation marks.
    '''
    response = llm.invoke(prompt)
    return response.content
def parse_title_generation(output):
    # Initialize a dictionary to store the parsed values
    parsed_data = {
        "TITLE": "",
        "EVALUATION": {
            "SCOPE INDICATION": "",
            "STAKEHOLDER FOCUS": "",
            "TIMEFRAME": "",
            "OUTCOME-ORIENTED": "",
            "KEYWORD OPTIMIZATION": "",
            "AVOID UNNECESSARY WORDS": "",
            "USE ACTIVE VOICE": "",
            "QUANTIFY IF POSSIBLE": "",
            "AVOID QUESTIONS": "",
            "BALANCE CREATIVITY AND CLARITY": "",
            "CONSISTENCY": "",
            "AVOID ABBREVIATIONS": "",
            "CLARITY AND SIMPLICITY": "",
            "ENGAGEMENT": "",
            "PRECISION": "",
            "LENGTH": "",
            "PERSPECTIVE": ""
        }
    }

    # Split the output by lines
    lines = output.split('\n')

    # Flag to indicate when we've reached the evaluation section
    evaluation_section = False

    for line in lines:
        line = line.strip()
        if line.startswith("TITLE:"):
            # Extract the title, removing quotation marks
            parsed_data["TITLE"] = line.split(":", 1)[1].strip().strip('"')
        elif line == "EVALUATION:":
            evaluation_section = True
        elif evaluation_section and ":" in line:
            # Split the line into key and value
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"')  # Remove quotation marks
            if key in parsed_data["EVALUATION"]:
                parsed_data["EVALUATION"][key] = value

    return parsed_data

def check_title(llm, title):
    guidelines = [
        "1. Scope indication: Includes a hint about the scale or scope of the problem.",
        "2. Stakeholder focus: Mentions key stakeholders affected by or involved in the problem.",
        "3. Timeframe: Indicates whether it's an urgent, ongoing, or future issue.",
        "4. Outcome-oriented: Suggests the desired result or improvement.",
        "5. Keyword optimization: Uses relevant keywords for searchability and categorization.",
        "6. Avoid unnecessary words: Eliminates articles and filler words when possible.",
        "7. Use active voice: Employs active rather than passive language for directness.",
        "8. Quantify if possible: Includes numbers or metrics if they add significant value.",
        "9. Avoid questions: Frames the title as a statement rather than a question.",
        "10. Balance creativity and clarity: Uses engaging language but prioritizes clarity over cleverness.",
        "11. Consistency: Ensures the title aligns with the content of the problem statement.",
        "12. Avoid abbreviations: Spells out terms unless universally recognized in the field.",
        "13. Clarity and Simplicity: Ensure the title is easy to understand and free of complex jargon unless necessary.",
        "14. Engagement: Make the title engaging to capture the reader's interest.",
        "15. Precision: Use precise and specific language to avoid vagueness.",
        "16. Length: Maintain a balance between brevity and informativeness, aiming for 5 to 12 words.",
        "17. Perspective: Reflect the perspective or approach being taken, such as policy, technology, or societal impact."
    ]

    prompt = f'''
    Evaluate the following title based on the given guidelines:

    TITLE: "{title}"

    Guidelines:
    {' '.join(guidelines)}

    Provide your evaluation in the following format:

    OVERALL_EVALUATION: "YES" or "NO"

    GUIDELINE_EVALUATIONS:
    SCOPE_INDICATION: "YES/NO"
    STAKEHOLDER_FOCUS: "YES/NO"
    TIMEFRAME: "YES/NO"
    OUTCOME_ORIENTED: "YES/NO"
    KEYWORD_OPTIMIZATION: "YES/NO"
    AVOID_UNNECESSARY_WORDS: "YES/NO"
    USE_ACTIVE_VOICE: "YES/NO"
    QUANTIFY_IF_POSSIBLE: "YES/NO"
    AVOID_QUESTIONS: "YES/NO"
    BALANCE_CREATIVITY_AND_CLARITY: "YES/NO"
    CONSISTENCY: "YES/NO"
    AVOID_ABBREVIATIONS: "YES/NO"
    CLARITY_AND_SIMPLICITY: "YES/NO"
    ENGAGEMENT: "YES/NO"
    PRECISION: "YES/NO"
    LENGTH: "YES/NO"
    PERSPECTIVE: "YES/NO"

    IMPROVEMENT_SUGGESTIONS: "Provide brief suggestions for improvement if the overall evaluation is NO. If YES, write 'No improvements needed.'"

    Note: Ensure all responses are within double quotes as shown in the format above.
    '''

    response = llm.invoke(prompt)
    return response.content

def parse_title_check(output):
    parsed_data = {
        "OVERALL_EVALUATION": "",
        "GUIDELINE_EVALUATIONS": {},
        "IMPROVEMENT_SUGGESTIONS": ""
    }

    lines = output.split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        if line.startswith("OVERALL_EVALUATION:"):
            parsed_data["OVERALL_EVALUATION"] = line.split(":", 1)[1].strip().strip('"')
        elif line == "GUIDELINE_EVALUATIONS:":
            current_section = "GUIDELINE_EVALUATIONS"
        elif line.startswith("IMPROVEMENT_SUGGESTIONS:"):
            parsed_data["IMPROVEMENT_SUGGESTIONS"] = line.split(":", 1)[1].strip().strip('"')
        elif current_section == "GUIDELINE_EVALUATIONS" and ":" in line:
            key, value = line.split(":", 1)
            parsed_data["GUIDELINE_EVALUATIONS"][key.strip()] = value.strip().strip('"')

    return parsed_data

def update_title(llm, title, feedback):
    prompt = f'''
    You are tasked with understanding the sentiment of feedback and updating a title if required based on the feedback provided by the user.
    
    Original Title: "{title}"
    
    Feedback: "{feedback}"
    
    Please respond in the following format:

    SENTIMENT: "POSITIVE" or "NEGATIVE"
    
    ACTION: "KEEP" or "UPDATE"
    
    TITLE: "The original or updated title goes here"
    
    EXPLANATION: "Brief explanation of why the title was kept or updated"

    Note: Ensure all responses are within double quotes as shown in the format above.
    '''
    
    response = llm.invoke(prompt)
    return response.content

def parse_title_update(output):
    parsed_data = {
        "SENTIMENT": "",
        "ACTION": "",
        "TITLE": "",
        "EXPLANATION": ""
    }

    lines = output.split('\n')

    for line in lines:
        line = line.strip()
        if line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"')
            if key in parsed_data:
                parsed_data[key] = value

    return parsed_data

def generate_abstract(llm, title, extracted_problem):
    prompt = f'''
    You are tasked with generating an abstract for a problem statement from information provided by the innovator.
    
    Title: "{title}"
    
    Problem Information: "{extracted_problem}"
    
    Using these elements, create an abstract following the given guidelines. 
    
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
    
    Respond in the following format:

    ABSTRACT: "Your generated abstract goes here"

    REASONING: "Provide a brief explanation of how the abstract addresses the key guidelines"

    Note: Ensure all responses are within double quotes as shown in the format above. 
    ABSTRACT response should be within double quotes. ABSTRACT: "......."
    REASONING response should be within double quotes. REASONING: "......."
    '''
    
    response = llm.invoke(prompt)
    return response.content

def parse_abstract_generation(output):
    parsed_data = {
        "ABSTRACT": "",
        "REASONING": ""
    }

    lines = output.split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith("ABSTRACT:"):
            parsed_data["ABSTRACT"] = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("REASONING:"):
            parsed_data["REASONING"] = line.split(":", 1)[1].strip().strip('"')

    return parsed_data

def update_abstract(llm, abstract, feedback):
    prompt = f'''
    You are tasked with understanding the sentiment of feedback and updating an abstract if required based on the feedback provided by the user.
    
    Original Abstract: "{abstract}"
    
    Feedback: "{feedback}"
    
    Please respond in the following format:

    SENTIMENT: "POSITIVE" or "NEGATIVE"
    
    ACTION: "KEEP" or "UPDATE"
    
    ABSTRACT: "The original or updated abstract goes here"
    
    EXPLANATION: "Brief explanation of why the abstract was kept or updated"

    Note: Ensure all responses are within double quotes as shown in the format above.
    '''
    
    response = llm.invoke(prompt)
    return response.content

def parse_abstract_update(output):
    parsed_data = {
        "SENTIMENT": "",
        "ACTION": "",
        "ABSTRACT": "",
        "EXPLANATION": ""
    }

    lines = output.split('\n')
    current_key = None

    for line in lines:
        line = line.strip()
        if line:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"')
                if key in parsed_data:
                    parsed_data[key] = value
                    current_key = key
            elif current_key:
                # If there's no colon, it's a continuation of the previous value
                parsed_data[current_key] += " " + line.strip('"')

    return parsed_data


def assess_problem(llm, extracted_problem):
    prompt = f'''
    Analyze the following problem description and provide a classification based on its complexity and predictability:

    Problem Description: "{extracted_problem}"

    **Instructions:**
    Analyze the text to determine the level of complexity and predictability:

    **1. Complexity Assessment (score from 1-10):**
    - Count the number of distinct concepts, stakeholders, or systems mentioned
    - Identify keywords indicating complexity (e.g., "interconnected", "multi-faceted", "systemic")
    - Assess the scope of the problem (local, organizational, global)
    - Consider the timeframe mentioned (short-term vs long-term)

    **2. Predictability Assessment (score from 1-10):**
    - Identify language suggesting certainty or uncertainty
    - Look for mentions of established procedures or novel approaches
    - Assess the availability of historical data or similar past experiences
    - Consider the number of unknown variables or potential outcomes mentioned

    **3. Classification:**
    - If Complexity > 7 and Predictability < 4: Classify as COMPLEX
    - If Complexity > 7 and Predictability > 7: Classify as COMPLICATED
    - If Complexity < 4 and Predictability < 4: Classify as WICKED
    - If Complexity < 4 and Predictability > 7: Classify as SIMPLE
    - For scores falling between these ranges, classify based on the nearest quadrant or consider a hybrid classification

    **4. Confidence Score:**
    - Calculate a confidence score based on how clearly the problem fits into a quadrant
    - If confidence is low, flag for human review

    Please respond in the following format:

    COMPLEXITY_SCORE: "Score from 1-10"
    COMPLEXITY_REASONING: "Brief explanation for the complexity score"

    PREDICTABILITY_SCORE: "Score from 1-10"
    PREDICTABILITY_REASONING: "Brief explanation for the predictability score"

    CLASSIFICATION: "COMPLEX" or "COMPLICATED" or "WICKED" or "SIMPLE"
    CLASSIFICATION_REASONING: "Explanation for why this classification was chosen"

    CONFIDENCE_SCORE: "Score from 1-10"
    CONFIDENCE_REASONING: "Explanation for the confidence score"

    Note: Ensure all responses are within double quotes as shown in the format above.
    '''
    
    response = llm.invoke(prompt)
    return response.content

def parse_problem_assessment(output):
    parsed_data = {
        "COMPLEXITY_SCORE": "",
        "COMPLEXITY_REASONING": "",
        "PREDICTABILITY_SCORE": "",
        "PREDICTABILITY_REASONING": "",
        "CLASSIFICATION": "",
        "CLASSIFICATION_REASONING": "",
        "CONFIDENCE_SCORE": "",
        "CONFIDENCE_REASONING": ""
    }

    lines = output.split('\n')
    current_key = None

    for line in lines:
        line = line.strip()
        if line:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"')
                if key in parsed_data:
                    parsed_data[key] = value
                    current_key = key
            elif current_key:
                # If there's no colon, it's a continuation of the previous value
                parsed_data[current_key] += " " + line.strip('"')

    return parsed_data

# Function to explain problem classification
def explain_problem_classification(llm, extracted_problem,problem_classification):
    prompt = f'''
    ## Problem Analysis
    
    **Problem Description:**
    {extracted_problem}

    **Problem classifications:**
    {problem_classification}

    **Instructions:**
    You are tasked to reason the problem classification based on the extracted problems and the below given instructions. Make sure you include all 4 comparison parameters  to reason this classification:

    **1. Complexity Assessment (score from 1-10):**
    - Count the number of distinct concepts, stakeholders, or systems mentioned
    - Identify keywords indicating complexity (e.g., "interconnected", "multi-faceted", "systemic")
    - Assess the scope of the problem (local, organizational, global)
    - Consider the timeframe mentioned (short-term vs long-term)

    **2. Predictability Assessment (score from 1-10):**
    - Identify language suggesting certainty or uncertainty
    - Look for mentions of established procedures or novel approaches
    - Assess the availability of historical data or similar past experiences
    - Consider the number of unknown variables or potential outcomes mentioned

    **3. Classification:**
    - If Complexity > 7 and Predictability < 4: Classify as COMPLEX
    - If Complexity > 7 and Predictability > 7: Classify as COMPLICATED
    - If Complexity < 4 and Predictability < 4: Classify as WICKED
    - If Complexity < 4 and Predictability > 7: Classify as SIMPLE
    - For scores falling between these ranges, classify based on the nearest quadrant or consider a hybrid classification

    **4. Confidence Score:**
    - Calculate a confidence score based on how clearly the problem fits into a quadrant
    - If confidence is low, flag for human review

    ## Problem Classification Output

    **Results:**
    - Reasons for the given problem classification are: 
    '''

    response = llm.invoke(prompt)
    return response.content

#Function to classify problems based on user input
def user_enhanced_problem_classification(llm, extracted_problem,complexity,predictability):
    prompt = f'''
    ## Problem Analysis
    
    **Problem Description:**
    {extracted_problem}

    ** User input complexity and predictabilty **:
    Complexity: {complexity}
    Predictabilty: {predictability}

    **Instructions:**
    You are tasked to classify the problem based on the complexity and predictability and the below given instructions:

    **1. Classification:**
    - If Complexity > 7 and Predictability < 4: Classify as COMPLEX
    - If Complexity > 7 and Predictability > 7: Classify as COMPLICATED
    - If Complexity < 4 and Predictability < 4: Classify as WICKED
    - If Complexity < 4 and Predictability > 7: Classify as SIMPLE
    - For scores falling between these ranges, classify based on the nearest quadrant or consider a hybrid classification

    MAKE SURE THE CLASSIFICATIONS ARE STRICTLY BASED ON THE COMPLEXITY AND PREDICTABILITY SCORES GIVEN BY THE USER ACCORDING TO THE RULES GIVEN ABOVE.
    
    ## Problem Classification Output

    **Results:**
    - Classification: 
    - Confidence Score:
    - Reasons for the given problem classification are: 
    '''

    response = llm.invoke(prompt)
    return response.content

def generate_assumptions(llm, extracted_problem):
    prompt = f'''
    You are tasked with generating assumptions for the following problem statement:

    {extracted_problem}

    Generate 4 assumptions that follow these guidelines:
    1. Clearly stated and justified
    2. Relevant to the problem and potential solutions
    3. Acknowledges uncertainties
    4. Includes both technical and social aspects

    For each assumption, provide:
    ASSUMPTION: "The assumption statement"
    RATIONALE: "The justification for this assumption"
    TYPE: "TECHNICAL" or "SOCIAL"
    UNCERTAINTY_LEVEL: "LOW", "MEDIUM", or "HIGH"

    Respond in the following format:

    ASSUMPTION_1: "Assumption statement 1"
    RATIONALE_1: "Rationale for assumption 1"
    TYPE_1: "TECHNICAL" or "SOCIAL"
    UNCERTAINTY_1: "LOW" or "MEDIUM" or "HIGH"

    ASSUMPTION_2: "Assumption statement 2"
    RATIONALE_2: "Rationale for assumption 2"
    TYPE_2: "TECHNICAL" or "SOCIAL"
    UNCERTAINTY_2: "LOW" or "MEDIUM" or "HIGH"

    ASSUMPTION_3: "Assumption statement 3"
    RATIONALE_3: "Rationale for assumption 3"
    TYPE_3: "TECHNICAL" or "SOCIAL"
    UNCERTAINTY_3: "LOW" or "MEDIUM" or "HIGH"

    ASSUMPTION_4: "Assumption statement 4"
    RATIONALE_4: "Rationale for assumption 4"
    TYPE_4: "TECHNICAL" or "SOCIAL"
    UNCERTAINTY_4: "LOW" or "MEDIUM" or "HIGH"

    Note: Ensure all responses are within double quotes as shown in the format above.
    '''
    
    response = llm.invoke(prompt)
    return response.content

def parse_assumptions(output):
    parsed_data = {}
    lines = output.split('\n')
    current_key = None
    assumption_count = 0

    for line in lines:
        line = line.strip()
        if line:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"')
                
                if key.startswith("ASSUMPTION"):
                    assumption_count += 1
                    parsed_data[f"assumption_{assumption_count}"] = {
                        "assumption": value,
                        "rationale": "",
                        "type": "",
                        "uncertainty": ""
                    }
                elif key.startswith("RATIONALE"):
                    parsed_data[f"assumption_{assumption_count}"]["rationale"] = value
                elif key.startswith("TYPE"):
                    parsed_data[f"assumption_{assumption_count}"]["type"] = value
                elif key.startswith("UNCERTAINTY"):
                    parsed_data[f"assumption_{assumption_count}"]["uncertainty"] = value
                
                current_key = key
            elif current_key:
                # If there's no colon, it's a continuation of the previous value
                if current_key.startswith("ASSUMPTION"):
                    parsed_data[f"assumption_{assumption_count}"]["assumption"] += " " + line.strip('"')
                elif current_key.startswith("RATIONALE"):
                    parsed_data[f"assumption_{assumption_count}"]["rationale"] += " " + line.strip('"')

    return parsed_data
    
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

        10. Finally the description should contain step by step breakdown of the problem and not just be a bunch of paragraphs. Include all these above points in the description.
        
        
    '''
    response = llm.invoke(prompt)
    return response.content

#Funtion to suggest Problem Breadth and Depth model
def suggest_pdb_model(llm, problem):
    prompt = f"""
    ## Instruction ##
    As an AI assistant, suggest an appropriate problem-solving model for the given problem. Choose from these options:
    
    - 5Ws and H (Who, What, Where, When, Why, How)
    - 5Ps (People, Process, Products, Programs, Performance)
    - 5Ms (Man, Machine, Material, Method, Measurement)
    - 5Es (Environment, Education, Engineering, Enforcement, Evaluation)
    - 4Ps (Product, Price, Place, Promotion)

    Analyze the problem considering:
    1. Context and core issue
    2. Primary sector and desired outcome
    3. Key stakeholders and their roles
    4. Available resources and data requirements
    5. Problem characteristics and model strengths
    6. Model complexity vs. problem intricacy
    7. Implementation feasibility
    8. Potential for actionable insights

    Provide your response in this exact format:
    SUGGESTED MODEL: "........."
    REASONING: "........."

    NOTE: VERY IMPORTANT BOTH THE SUGGESTED MODEL and REASONING responses should be within double quotation "..." 

    Here are three examples:

    Example 1:
    Problem: A local newspaper wants to improve its reporting on community events.
    SUGGESTED MODEL: "5Ws and H"
    REASONING: "Ideal for journalistic inquiry, covering all aspects of an event comprehensively and structuring reporting effectively."

    Example 2:
    Problem: A manufacturing plant is experiencing high defect rates in its production line.
    SUGGESTED MODEL: "5Ms"
    REASONING: "Tailored for manufacturing, systematically analyzes all production aspects to identify sources of defects."

    Example 3:
    Problem: A non-profit organization aims to reduce teenage smoking in their city.
    SUGGESTED MODEL: "5Es"
    REASONING: "Comprehensive approach for public health initiatives, addressing multiple aspects of behavior change and policy implementation."

    Now, analyze this problem and suggest an appropriate model:

    Problem: {problem}

    
    NOTE: VERY IMPORTANT BOTH THE SUGGESTED MODEL and REASONING responses should be within double quotation "..."

    ## Response ##
    """
    
    response = llm.invoke(prompt)
    return response.content

def parse_pbd_suggestion(output):
    # Initialize a dictionary to store the parsed values
    parsed_data = {
        "SUGGESTED MODEL": "",
        "REASONING": ""
    }

    # Split the output by lines and process each line
    lines = output.split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith("SUGGESTED MODEL:"):
            parsed_data["SUGGESTED MODEL"] = line.split(":")[1].strip().strip('"')
        elif line.startswith("REASONING:"):
            parsed_data["REASONING"] = line.split(":")[1].strip().strip('"')
        elif "REASONING" in parsed_data and parsed_data["REASONING"]:
            parsed_data["REASONING"] += " " + line.strip().strip('"')

    return parsed_data

def analyze_with_5w1h(llm, problem):
    prompt = f"""
    ## Instruction ##
    As an AI analyst, use the 5Ws and H model (Who, What, Where, When, Why, How) to thoroughly analyze the given problem. This model is versatile and applicable across various domains including journalism, business analysis, project management, and general problem-solving.

    Before answering each question, consider the following:

    1. Context: Understand the broader context of the problem, including industry trends, historical background, and current landscape.
    2. Stakeholders: Identify all parties involved or affected by the problem.
    3. Data: Consider what data is available or needed to support your analysis.
    4. Implications: Think about the potential consequences and impacts of the problem and its solution.
    5. Interconnections: Explore how the different aspects of the problem relate to each other.
    6. Time frame: Consider both short-term and long-term perspectives.
    7. Scale: Assess the scope of the problem - is it local, global, or somewhere in between?
    8. Resources: Think about the resources available or required to address the problem.
    9. Constraints: Identify any limitations or restrictions that may affect the situation.
    10. Opportunities: Look for potential positive outcomes or benefits that could arise.

    For each of the 5Ws and H, provide a concise but comprehensive answer. Your analysis should be thorough, considering multiple angles and possibilities.

    Problem: {problem}

    Provide your analysis in the following format:

    WHO: "......"
    - Key stakeholders:
    - Affected parties:

    WHAT: "......"
    - Core issue:
    - Related factors:

    WHERE: "......"
    - Physical location(s):
    - Contextual environment:

    WHEN: "......"
    - Timeframe:
    - Relevant deadlines or milestones:

    WHY: "......"
    - Root causes:
    - Motivating factors:

    HOW: "......"
    - Potential solutions:
    - Implementation challenges:

    SUMMARY:
    "Provide a brief summary of your analysis, highlighting the most critical aspects and any key insights or recommendations."

    ## Response ##
    """
    
    response = llm.invoke(prompt)
    return response.content

def analyze_with_5ps(llm, problem):
    prompt = f"""
    ## Instruction ##
    As an AI business analyst, use the 5Ps model (People, Process, Products, Programs, Performance) to comprehensively analyze the given organizational problem or situation. This framework is crucial for strategic decision-making and organizational development.

    Before analyzing each P, consider the following:

    1. Industry context: Understand current market trends, competitive landscape, and industry best practices.
    2. Organizational culture: Consider the company's values, norms, and working environment.
    3. Resource allocation: Evaluate how resources are distributed across different areas.
    4. Technology integration: Assess the role of technology in each aspect of the organization.
    5. Stakeholder expectations: Consider the needs and expectations of all stakeholders.
    6. Regulatory environment: Be aware of relevant laws, regulations, and compliance requirements.
    7. Long-term strategy: Align your analysis with the organization's long-term goals and vision.
    8. Change management: Consider the organization's ability to adapt and implement changes.
    9. Measurement and metrics: Identify key performance indicators for each area.
    10. Interdependencies: Explore how each P impacts and interacts with the others.

    For each of the 5Ps, provide a concise but comprehensive analysis. Your response should be thorough, considering multiple facets of the organization.

    Problem: {problem}

    Provide your analysis in the following format:

    PEOPLE: "......"
    - Key personnel and roles:
    - Skills and competencies:
    - Organizational structure:

    PROCESS: "......"
    - Core business processes:
    - Efficiency and bottlenecks:
    - Process integration:

    PRODUCTS: "......"
    - Product/service portfolio:
    - Market positioning:
    - Innovation pipeline:

    PROGRAMS: "......"
    - Key initiatives and projects:
    - Resource allocation:
    - Program effectiveness:

    PERFORMANCE: "......"
    - Key performance indicators:
    - Benchmarking results:
    - Areas for improvement:

    STRATEGIC IMPLICATIONS:
    "Provide a brief summary of your analysis, highlighting the most critical aspects across the 5Ps. Offer key insights and strategic recommendations based on your holistic review."

    ## Response ##
    """
    
    response = llm.invoke(prompt)
    return response.content

def analyze_with_5ms(llm, problem):
    prompt = f"""
    ## Instruction ##
    As an AI manufacturing and quality control analyst, use the 5Ms model (Man, Machine, Material, Method, Measurement) to thoroughly analyze the given production or quality control problem. This framework is crucial for identifying root causes of issues and improving manufacturing processes.

    Before analyzing each M, consider the following:

    1. Industry standards: Understand current manufacturing best practices and quality benchmarks.
    2. Regulatory compliance: Be aware of relevant manufacturing and safety regulations.
    3. Lean manufacturing principles: Consider how lean concepts apply to the situation.
    4. Supply chain dynamics: Evaluate how upstream and downstream processes impact the problem.
    5. Technology trends: Assess the role of emerging technologies in manufacturing and quality control.
    6. Environmental factors: Consider environmental impacts and sustainability concerns.
    7. Cost implications: Analyze the financial aspects of the problem and potential solutions.
    8. Scalability: Consider how solutions can be scaled across production lines or facilities.
    9. Continuous improvement: Think about how ongoing monitoring and improvement can be implemented.
    10. Cross-functional impacts: Evaluate how changes in one area might affect others.

    For each of the 5Ms, provide a concise but comprehensive analysis. Your response should be thorough, considering multiple aspects of the manufacturing process.

    Problem: {problem}

    Provide your analysis in the following format:

    MAN: "......"
    - Workforce skills and training:
    - Human factors and ergonomics:
    - Shift patterns and fatigue management:

    MACHINE: "......"
    - Equipment capabilities and limitations:
    - Maintenance schedules and issues:
    - Automation and technology integration:

    MATERIAL: "......"
    - Raw material quality and consistency:
    - Inventory management:
    - Material handling and storage:

    METHOD: "......"
    - Production processes and workflows:
    - Standard operating procedures:
    - Process optimization opportunities:

    MEASUREMENT: "......"
    - Quality control metrics:
    - Inspection methods and frequency:
    - Data collection and analysis techniques:

    ROOT CAUSE ANALYSIS AND RECOMMENDATIONS:
    "Provide a brief summary of your analysis, identifying the most likely root causes of the problem across the 5Ms. Offer key insights and specific recommendations for process improvement and quality enhancement."

    ## Response ##
    """
    
    response = llm.invoke(prompt)
    return response.content

def analyze_with_5es(llm, problem):
    prompt = f"""
    ## Instruction ##
    As an AI policy and program analyst, use the 5Es model (Environment, Education, Engineering, Enforcement, Evaluation) to comprehensively analyze the given public policy, safety program, or behavior change initiative. This framework is crucial for developing, implementing, and assessing effective interventions.

    Before analyzing each E, consider the following:

    1. Stakeholder landscape: Identify all groups affected by or influencing the policy/program.
    2. Cultural context: Consider societal norms, values, and cultural factors.
    3. Resource availability: Assess available and required resources for implementation.
    4. Political climate: Understand the political factors that may impact the initiative.
    5. Long-term sustainability: Consider the long-term viability and impact of interventions.
    6. Ethical implications: Evaluate potential ethical issues or dilemmas.
    7. Interdisciplinary approach: Consider insights from relevant fields (e.g., psychology, sociology, economics).
    8. Technological integration: Assess how technology can support or hinder the initiative.
    9. Scalability: Consider how the program can be scaled up or adapted to different contexts.
    10. Unintended consequences: Anticipate potential negative outcomes or side effects.

    For each of the 5Es, provide a concise but comprehensive analysis. Your response should be thorough, considering multiple aspects of policy implementation and behavior change.

    Problem: {problem}

    Provide your analysis in the following format:

    ENVIRONMENT: "......"
    - Physical and social context:
    - Existing policies and infrastructure:
    - Barriers and facilitators:

    EDUCATION: "......"
    - Target audience and key messages:
    - Educational strategies and channels:
    - Awareness and knowledge gaps:

    ENGINEERING: "......"
    - Design interventions and modifications:
    - Technological solutions:
    - Infrastructure improvements:

    ENFORCEMENT: "......"
    - Regulatory measures:
    - Compliance strategies:
    - Incentives and disincentives:

    EVALUATION: "......"
    - Key performance indicators:
    - Monitoring methods:
    - Feedback mechanisms:

    STRATEGIC RECOMMENDATIONS:
    "Provide a brief summary of your analysis, highlighting the most critical aspects across the 5Es. Offer key insights and specific recommendations for policy design, implementation, and assessment. Consider short-term actions and long-term strategies."

    ## Response ##
    """
    
    response = llm.invoke(prompt)
    return response.content

def analyze_with_4ps(llm, problem):
    prompt = f"""
    ## Instruction ##
    As an AI marketing strategist, use the 4Ps model (Product, Price, Place, Promotion) to comprehensively analyze the given marketing challenge or opportunity. This Marketing Mix framework is crucial for developing effective marketing strategies and bringing products or services to market successfully.

    Before analyzing each P, consider the following:

    1. Target audience: Clearly define and understand the target market segments.
    2. Competitive landscape: Analyze competitors' strategies and market positioning.
    3. Brand identity: Consider how the strategy aligns with overall brand image and values.
    4. Market trends: Identify current and emerging trends in the industry.
    5. Customer journey: Map out the customer's path to purchase and key touchpoints.
    6. Digital transformation: Evaluate the role of digital channels and technologies.
    7. Regulatory environment: Be aware of relevant marketing and advertising regulations.
    8. Sustainability: Consider environmental and social responsibility aspects.
    9. Global vs. local approach: Assess the need for market-specific adaptations.
    10. ROI and metrics: Identify key performance indicators for marketing success.

    For each of the 4Ps, provide a concise but comprehensive analysis. Your response should be thorough, considering multiple aspects of marketing strategy.

    Problem: {problem}

    Provide your analysis in the following format:

    PRODUCT: "......"
    - Core features and benefits:
    - Product line and portfolio:
    - Branding and packaging:

    PRICE: "......"
    - Pricing strategy and positioning:
    - Discount and promotion policies:
    - Payment terms and options:

    PLACE: "......"
    - Distribution channels:
    - Market coverage:
    - Inventory and logistics:

    PROMOTION: "......"
    - Marketing communication mix:
    - Key messages and unique selling propositions:
    - Media strategy and budget allocation:

    INTEGRATED MARKETING STRATEGY:
    "Provide a brief summary of your analysis, highlighting how the 4Ps interact and support each other. Offer key insights and specific recommendations for an integrated marketing approach. Consider both short-term tactics and long-term strategic positioning."

    ## Response ##
    """
    
    response = llm.invoke(prompt)
    return response.content


# Function to generate problem breadth and depth
def generate_breadth_and_depth(llm,extracted_problems):
    prompt = f"""
    You are provided with  a detailed description of the problem. Use these details to generate the problem breadth and depth by answering the 5Ws and 1H.

    *Problem Description:*
    {extracted_problems}

    Your task is to provide a detailed response to the following questions:
    **What:** Define the Problem Statement. This is the type of question we ask in order to narrow the problem and focus in on key issues.
    **When:** Clearly identifying the time related aspects of the problem. When does the conflict occur? Is the key question here.
    **Where:** The 'Where?' key is relating to the ‘zones of conflict’. Determine what is the zone of conflict looking at the super-system, system and sub-system.
    **Who:** Clearly identify the person connected with the problem. He could be one who is using the final product or anyone in the line-up of concept-to-market or a person at any of the product Life-stages.
    **How:** The how question is present to encourage you to think about the underlying causes and effects of the problem. How does the conflict arise?

    Now think about these inverse questions:
    **What is not a Problem?**: In contrast to the above context of "what is the problem", identify what is not a problem in the current scenario.
    **When is it not a Problem?**: When is it not a problem or when is the time the conflict doesn't occur?
    **Where is it not a Problem?**: Identify the zones of conflict where the problem doesn't occur looking at the super-system, system, and sub-system.
    **Who is not affected?**: Here, identify the people who are not connected with the problem. They could be ones who are not using the final product or anyone not affected by the problem.
    **How is it not a Problem?**: The how-not question is present to encourage us to think about how it is not affecting the current environment.

    Provide comprehensive and relevant answers based on the given domain, sub-domain, title, abstract, and description.
    """
    response = llm.invoke(prompt)
    return response.content

# Function to update problem depth and breadth
def update_depth_breadth(llm,problem_breadth_depth,feedback):
    prompt = f"""
    You are tasked with understand the sentiment of a feedback and updating the problem breadth and depth if required based on the feedback provided by the user.
        Here is the original Title:
        {problem_breadth_depth}
        
        And here is the feedback:
        {feedback}
        
        If the feedback is positive then your response will the same problem breadth and depth, otherwise please update the problem breadth and depth according to the feedback. 
        Thus your response will be one of the following:

        Output:

        Okay then we will stick with the same Title:
        {problem_breadth_depth}

        or

        Okay here's an updated Problem breadth and depth:
        
    """
    response = llm.invoke(prompt)
    return response.content


def problem_landscape(llm, extracted_information):
    prompt = f"""
        ## CONTEXT ##
        Here is some information about the problem:
        {extracted_information}

        ## TASK ##
        Your task is to create a detailed function map for the given technology/system. Identify specific past, present, and future systems, subsystems, and supersystems. A function map illustrates the relationships between systems, subsystems, and super-systems within a larger framework.

        Follow these guidelines:

        1. **Supersystem**:
            - **Past**: Identify specific broader applications, industries, and environments where the technology/system was used historically.
            - **Present**: List current specific broader applications, industries, and environments.
            - **Future**: Predict specific future broader applications, industries, and environments, including emerging trends.

        2. **System**:
            - **Past**: Name specific architectures, designs, or solutions used historically. Include actual technologies or products when possible.
            - **Present**: Describe current specific architectures, designs, or solutions. Focus on the main technology in question.
            - **Future**: Predict specific future architectures, designs, or solutions. Include emerging technologies and theoretical concepts.

        3. **Subsystem**:
            - **Past**: List specific critical components, technologies, and materials used historically.
            - **Present**: Enumerate current specific critical components, technologies, and materials.
            - **Future**: Predict specific future critical components, technologies, and materials. Include cutting-edge research areas.

        Aim for high specificity and technical detail in your responses. Include actual technology names, scientific concepts, and industry-specific terminology where appropriate. Provide 4-5 examples for each category when possible.

        Avoid using any kind of Brand name in any of your response. Always use actual technology names, scientific concepts, and industry-specific terminology.

        Use the following format for your response:

        PAST SUPER SYSTEM: "..."
        PAST SYSTEM: "..."
        PAST SUB SYSTEM: "..."

        PRESENT SUPER SYSTEM: "..."
        PRESENT SYSTEM: "..."
        PRESENT SUB SYSTEM: "..."

        FUTURE SUPER SYSTEM: "..."
        FUTURE SYSTEM: "..."
        FUTURE SUB SYSTEM: "..."

        ##EXAMPLE INPUT##
        Lithium-ion batteries
        ## EXAMPLE OUTPUT ##
        PAST SUPER SYSTEM: "Early battery technologies, Electrical systems, Automotive industry, Portable electronic devices"
        PAST SYSTEM: "Lead-acid batteries, Nickel-cadmium batteries, Nickel-metal hydride batteries, Lithium-ion batteries, Zinc-carbon batteries"
        PAST SUB SYSTEM: "Electrodes, Electrolytes, Battery management circuits, Casing and packaging"

        PRESENT SUPER SYSTEM: "Renewable energy systems, Electric vehicles, Consumer electronics, Industrial power backup systems"
        PRESENT SYSTEM: "Lithium-ion batteries"
        PRESENT SUB SYSTEM: "Battery cells, Battery management system (BMS), Thermal management system, Charging system"

        FUTURE SUPER SYSTEM: "Smart grid energy storage, High-capacity energy storage, Advanced transportation systems, Wearable technology, Space exploration technology"
        FUTURE SYSTEM: "Solid-state batteries, Graphene-based batteries, Sodium-ion batteries, Flow batteries, Aluminium-air batteries"
        FUTURE SUB SYSTEM: "Advanced electrode materials, Energy-dense electrolytes, Intelligent BMS, Rapid charging technology, Self-healing battery technologies"

        Please provide highly detailed and specific components for each category in a similar format for the problem system: {extracted_information}.
        
        Your Input: {extracted_information}
        Your output:

    """
    response = llm.invoke(prompt)
    return response.content

#Function 
def parse_problem_landscape_output(output):
    # Initialize a dictionary to store the parsed values
    parsed_data = {
        "past_super_system": "",
        "past_system": "",
        "past_sub_system": "",
        "present_super_system": "",
        "present_system": "",
        "present_sub_system": "",
        "future_super_system": "",
        "future_system": "",
        "future_sub_system": ""
    }

    # Split the output by lines and process each line
    for line in output.split('\n'):
        # Remove leading and trailing whitespace from each line
        line = line.strip()
        if line.startswith("PAST SUPER SYSTEM:"):
            parsed_data["past_super_system"] = line.split(":")[1].strip().strip('"')
        elif line.startswith("PAST SYSTEM:"):
            parsed_data["past_system"] = line.split(":")[1].strip().strip('"')
        elif line.startswith("PAST SUB SYSTEM:"):
            parsed_data["past_sub_system"] = line.split(":")[1].strip().strip('"')
        elif line.startswith("PRESENT SUPER SYSTEM:"):
            parsed_data["present_super_system"] = line.split(":")[1].strip().strip('"')
        elif line.startswith("PRESENT SYSTEM:"):
            parsed_data["present_system"] = line.split(":")[1].strip().strip('"')
        elif line.startswith("PRESENT SUB SYSTEM:"):
            parsed_data["present_sub_system"] = line.split(":")[1].strip().strip('"')
        elif line.startswith("FUTURE SUPER SYSTEM:"):
            parsed_data["future_super_system"] = line.split(":")[1].strip().strip('"')
        elif line.startswith("FUTURE SYSTEM:"):
            parsed_data["future_system"] = line.split(":")[1].strip().strip('"')
        elif line.startswith("FUTURE SUB SYSTEM:"):
            parsed_data["future_sub_system"] = line.split(":")[1].strip().strip('"')
    #st.write(parsed_data)
    # Create a DataFrame
    data = {
        "Past": [parsed_data["past_super_system"], parsed_data["past_system"], parsed_data["past_sub_system"]],
        "Present": [parsed_data["present_super_system"], parsed_data["present_system"], parsed_data["present_sub_system"]],
        "Future": [parsed_data["future_super_system"], parsed_data["future_system"], parsed_data["future_sub_system"]]
    }

    index = ["Super System", "System", "Sub System"]

    df = pd.DataFrame(data, index=index)

    return df,parsed_data

#Funtions to create function map updated.
def identify_useful_function_map(llm, components):
    prompt = f"""
    Create a detailed Useful Function Map for the given system components. Follow these steps:

    1. Use the provided components to create a matrix where both rows and columns are labeled with these components.
    2. For each interaction between components, identify only the useful functions (UF). Think very carefully about all possible useful functions.
    3. Describe how one component positively affects the other using concise phrases that include both the action and the affected component.
    4. Place these functions in the corresponding cell of the matrix.

    Guidelines:
    - Use "UF:" to prefix useful functions.
    - Be specific and accurate when identifying functions. Include both the action and the affected component.
    - If there's no useful interaction between components, leave the cell empty or use a dash (-).
    - Multiple functions in the same cell should be separated by a line break.

    Here's an example of the desired output format based on the provided CSV data:

    | Components | Three point linkage | Blades | Chain and cover | Water | Stone | Mud | Shaft | Tiller cover | Tractor | Farmer |
    |------------|---------------------|--------|-----------------|-------|-------|-----|-------|--------------|---------|--------|
    | Three point linkage | - | - | - | - | - | - | - | - | UF: Pull tractor | UF: Farmer connects |
    | Blades | - | - | - | - | - | UF: Cuts mud | UF: Shaft rotates blades | - | - | - |
    | Chain and cover | - | - | - | - | - | - | UF: Rotates shaft | - | - | - |
    | Water | - | - | - | - | - | - | - | UF: Tiller cover stops water | - | - |
    | Stone | - | - | - | - | - | - | - | UF: Tiller cover protects from stone | - | - |
    | Mud | - | UF: Cuts blades | - | - | - | - | - | UF: Tiller cover stops mud | - | - |
    | Shaft | - | UF: Rotates blades | UF: Rotates chain | - | - | - | - | - | - | - |
    | Tiller cover | - | - | - | UF: Stops water | UF: Protects from stone | UF: Stops mud | - | - | - | - |
    | Tractor | UF: Pulls three point linkage | - | - | - | - | - | - | - | - | UF: Farmer drives tractor |
    | Farmer | UF: Connects three point linkage | - | - | - | - | - | - | - | UF: Drives tractor | - |

    This example demonstrates the ideal format and level of detail we're aiming for. Your task is to create a similar matrix using the provided components, identifying only useful functions for each interaction.

    Components: {components}

    Please provide the Useful Function Map in a markdown table format similar to the example above, using the given components.
    """
    response = llm.invoke(prompt)
    return response.content

def identify_harmful_function_map(llm, components):
    prompt = f"""
    Create a detailed Harmful Function Map for the given system components. Follow these steps:

    1. Use the provided components to create a matrix where both rows and columns are labeled with these components.
    2. For each interaction between components, identify only the harmful functions (HF). Think very carefully about all possible harmful functions.
    3. Describe how one component negatively affects the other using concise phrases that include both the action and the affected component.
    4. Place these functions in the corresponding cell of the matrix.

    Guidelines:
    - Use "HF:" to prefix harmful functions.
    - Be specific and accurate when identifying functions. Include both the action and the affected component.
    - If there's no harmful interaction between components, leave the cell empty or use a dash (-).
    - Multiple functions in the same cell should be separated by a line break.

    Here's an example of the desired output format based on the provided CSV data:

    | Components | Three point linkage | Blades | Chain and cover | Water | Stone | Mud | Shaft | Tiller cover | Tractor | Farmer |
    |------------|---------------------|--------|-----------------|-------|-------|-----|-------|--------------|---------|--------|
    | Three point linkage | - | - | - | - | - | - | - | - | - | - |
    | Blades | - | - | - | HF: Water rusts blades | HF: Stone breaks blades | HF: Throws mud | - | - | - | - |
    | Chain and cover | - | - | - | HF: Water rusts chain and cover | - | - | - | - | - | - |
    | Water | - | HF: Rusts blades | HF: Rusts chain and cover | - | - | - | HF: Rusts shaft | HF: Rusts tiller cover | - | - |
    | Stone | - | HF: Breaks blades | - | - | - | - | HF: Hits shaft | HF: Hits tiller cover | - | - |
    | Mud | - | HF: Blade throws mud | - | - | - | - | HF: Sticks to shaft | - | - | - |
    | Shaft | - | - | - | HF: Water rusts shaft | HF: Stone hits shaft | HF: Mud sticks to shaft | - | - | - | - |
    | Tiller cover | - | - | - | HF: Water rusts tiller cover | HF: Stone hits tiller cover | - | - | - | - | - |
    | Tractor | - | - | - | - | - | - | - | - | - | - |
    | Farmer | - | - | - | - | - | - | - | - | - | - |

    This example demonstrates the ideal format and level of detail we're aiming for. Your task is to create a similar matrix using the provided components, identifying only harmful functions for each interaction.

    Components: {components}

    Please provide the Harmful Function Map in a markdown table format similar to the example above, using the given components.
    """
    response = llm.invoke(prompt)
    return response.content

#Function to use CREATE think model for Ideas.
def create_adjacent_domain_prompt(llm, idea):
    prompt = f"""
    Let's explore ways to enhance and reimagine your product idea using the CREATE method. We'll go through each step together, considering both intuitive insights and creative suggestions.

    WHEN YOU ARE FRAMING YOUR ANSWER THINK OF ADJACENTS DOMAIN OF THE FOLLOWING IDEA AND USE DATA FROM ADJACENT DOMAINS TO ANSWER THE GIVEN POINTS.
    
    Your product idea:
    {idea}

    For each CREATE step, we'll consider:
    1. Intuitive insights: What naturally comes to mind when thinking about this aspect?
    2. Creative suggestions: Some fresh ideas to consider.

    Let's begin our journey through CREATE:

    1. Combine
       How might we blend this with other concepts or technologies for added value?

    2. Reverse
       What if we flipped our approach or looked at this from a new angle?

    3. Eliminate
       Are there aspects we could simplify or remove to streamline the product?

    4. Adapt
       In what ways could we adjust the product for different uses or markets?

    5. Transform (Modify)
       What changes in form or function might enhance the product?

    6. Explore Other Uses
       What unexpected applications might we discover for this product?

    Feel free to let your imagination run wild - there are no wrong answers here. Let's see where this creative journey takes us!

    Please note:
    The Creative suggestions for every part of the response should be based on these intuitive insights given the same part.

    Please provide your thoughts on each CREATE step.
    WHEN YOU ARE FRAMING YOUR ANSWER THINK OF ADJACENTS DOMAIN OF THE FOLLOWING IDEA AND USE DATA FROM ADJACENT DOMAINS TO ANSWER THE GIVEN POINTS.

    Here is an example input: Electric Vehicle Battery
    Here is an example output:
    Sure, let's integrate insights and suggestions from adjacent domains to further enrich our CREATE analysis for EV batteries. We'll draw from areas such as renewable energy, smart technology, aerospace, healthcare, and consumer electronics.

    ### 1. Combine:
    - **Intuitive insights:**
    - Combining EV batteries with renewable energy sources like solar panels or wind turbines for charging.
    - Integrating smart technology for real-time monitoring and optimization.
    - Combining EV batteries with energy storage systems for homes.

    - **Creative suggestions:**
    - **From Renewable Energy:** Integrate EV batteries with microgrids to enhance energy resilience in communities.
    - **From Smart Technology:** Embed IoT sensors to create a network of connected batteries that can share performance data for collective optimization.
    - **From Aerospace:** Use lightweight composite materials to reduce battery weight while maintaining durability.
    - **From Healthcare:** Apply wearable tech principles to develop flexible battery components that conform to the vehicle’s shape.

    ### 2. Reverse:
    - **Intuitive insights:**
    - Considering the recycling process first to ensure the entire lifecycle of the battery is sustainable.
    - Exploring the potential of second-life applications for used EV batteries.
    - Developing a battery that can be charged in reverse, i.e., it can supply power back to the grid or other devices.

    - **Creative suggestions:**
    - **From Circular Economy:** Implement closed-loop recycling systems where materials from old batteries are reused in new ones.
    - **From Consumer Electronics:** Adopt modular design practices, like those in smartphones, to allow for easy upgrades and replacements.
    - **From Data Centers:** Utilize liquid cooling systems to enhance thermal management and improve battery longevity.

    ### 3. Eliminate:
    - **Intuitive insights:**
    - Removing heavy metals and toxic materials to make the battery more environmentally friendly.
    - Simplifying the design to reduce manufacturing complexity and costs.
    - Eliminating rare and expensive materials by finding alternative resources.

    - **Creative suggestions:**
    - **From Green Chemistry:** Use green solvents and environmentally benign materials in battery production.
    - **From Automotive Industry:** Adopt lean manufacturing principles to minimize waste and streamline production.
    - **From Biotechnology:** Explore the use of biopolymers as sustainable alternatives to traditional battery materials.

    ### 4. Adapt:
    - **Intuitive insights:**
    - Adjusting battery capacity for different vehicle types, from scooters to trucks.
    - Customizing battery shapes and sizes to fit various vehicle designs.
    - Adapting the battery for extreme weather conditions.

    - **Creative suggestions:**
    - **From Construction:** Develop ruggedized batteries designed to withstand harsh environments, similar to those used in heavy machinery.
    - **From Textile Industry:** Create flexible, fabric-like batteries that can be integrated into various vehicle surfaces.
    - **From Space Exploration:** Implement radiation-hardened components to ensure battery reliability in extreme conditions.

    ### 5. Transform:
    - **Intuitive insights:**
    - Enhancing battery energy density and longevity.
    - Improving charging speed and efficiency.
    - Transforming the battery’s form factor for better integration into vehicle designs.

    - **Creative suggestions:**
    - **From Consumer Electronics:** Use solid-state technology to create thinner, more energy-dense batteries.
    - **From Wearable Tech:** Develop batteries with energy-harvesting capabilities to recharge from ambient sources like sunlight or motion.
    - **From Robotics:** Implement autonomous self-repair mechanisms that enable the battery to fix minor damages on the go.

    ### 6. Explore Other Uses:
    - **Intuitive insights:**
    - Using EV batteries for grid storage when they are no longer efficient for vehicle use.
    - Employing old EV batteries in renewable energy projects.
    - Utilizing EV batteries in portable power solutions.

    - **Creative suggestions:**
    - **From Agriculture:** Adapt EV batteries for use in electric tractors and farm equipment, supporting sustainable agriculture.
    - **From Home Automation:** Integrate batteries into smart home systems, providing backup power and enhancing energy efficiency.
    - **From Public Transportation:** Use repurposed EV batteries to power electric buses and trains, reducing urban pollution.

    ### Summary:

    - **Combine:**
    - Intuitive insights: Integrating with renewable energy, smart technology, and home energy systems.
    - Creative suggestions: Microgrid integration, IoT connectivity, lightweight composites, flexible components.

    - **Reverse:**
    - Intuitive insights: Recycling, second-life applications, reversible charging.
    - Creative suggestions: Closed-loop recycling, modular designs, liquid cooling systems.

    - **Eliminate:**
    - Intuitive insights: Removing harmful materials, simplifying design, finding alternative resources.
    - Creative suggestions: Green solvents, lean manufacturing, biopolymers.

    - **Adapt:**
    - Intuitive insights: Adjusting capacity, customizing shapes, adapting for weather conditions.
    - Creative suggestions: Ruggedized designs, flexible fabric-like batteries, radiation-hardened components.

    - **Transform:**
    - Intuitive insights: Improving density, charging speed, and form factor.
    - Creative suggestions: Solid-state technology, energy-harvesting, autonomous self-repair.

    - **Explore Other Uses:**
    - Intuitive insights: Grid storage, renewable energy projects, portable power.
    - Creative suggestions: Electric farm equipment, smart home integration, public transportation applications.

    ##Your input : {idea}
    Your output:

    Give output in markdown format.
    
    
    """
    response = llm.invoke(prompt)
    return response.content

#Function to use CREATE think model for Ideas.
def create_same_domain_prompt(llm, idea):
    prompt = f"""
    Let's explore ways to enhance and reimagine your product idea using the CREATE method. We'll go through each step together, considering both intuitive insights and creative suggestions.

    Your product idea:
    {idea}

    For each CREATE step, we'll consider:
    1. Intuitive insights: What naturally comes to mind when thinking about this aspect?
    2. Creative suggestions: Some fresh ideas to consider.

    Let's begin our journey through CREATE:

    1. Combine
       How might we blend this with other concepts or technologies for added value?

    2. Reverse
       What if we flipped our approach or looked at this from a new angle?

    3. Eliminate
       Are there aspects we could simplify or remove to streamline the product?

    4. Adapt
       In what ways could we adjust the product for different uses or markets?

    5. Transform (Modify)
       What changes in form or function might enhance the product?

    6. Explore Other Uses
       What unexpected applications might we discover for this product?

    Feel free to let your imagination run wild - there are no wrong answers here. Let's see where this creative journey takes us!

    Please note:
    The Creative suggestions for every part of the response should be based on these intuitive insights given the same part.

    Please provide your thoughts on each CREATE step. Here is an examples: 

    Input: "Electric Vehicle Battery"
    Output:
    "Sure, let's dive into the CREATE method to explore and enhance the idea of an electric vehicle (EV) battery.

    ### 1. Combine:
    - **Intuitive insights:**
    - Combining EV batteries with renewable energy sources like solar panels or wind turbines for charging.
    - Integrating smart technology for real-time monitoring and optimization.
    - Combining EV batteries with energy storage systems for homes.

    - **Creative suggestions:**
    - Develop a hybrid battery system that uses both solid-state and liquid electrolyte components for increased safety and performance.
    - Integrate AI-driven predictive maintenance to forecast and address potential battery issues before they occur.
    - Create a modular battery system that can be easily swapped or upgraded without replacing the entire unit.

    ### 2. Reverse:
    - **Intuitive insights:**
    - Considering the recycling process first to ensure the entire lifecycle of the battery is sustainable.
    - Exploring the potential of second-life applications for used EV batteries.
    - Developing a battery that can be charged in reverse, i.e., it can supply power back to the grid or other devices.

    - **Creative suggestions:**
    - Design a battery that prioritizes disassembly and recycling, with components that are easier to separate and repurpose.
    - Implement a leasing model where consumers lease batteries, and manufacturers take responsibility for end-of-life recycling.
    - Create a dual-function battery that not only powers the vehicle but can also serve as a mobile power bank for external devices.

    ### 3. Eliminate:
    - **Intuitive insights:**
    - Removing heavy metals and toxic materials to make the battery more environmentally friendly.
    - Simplifying the design to reduce manufacturing complexity and costs.
    - Eliminating rare and expensive materials by finding alternative resources.

    - **Creative suggestions:**
    - Develop a battery using bio-based or organic materials that are both sustainable and less harmful to the environment.
    - Streamline the battery management system to reduce the number of electronic components, making the battery lighter and more efficient.
    - Create a simplified manufacturing process using 3D printing technology to reduce waste and lower production costs.

    ### 4. Adapt:
    - **Intuitive insights:**
    - Adjusting battery capacity for different vehicle types, from scooters to trucks.
    - Customizing battery shapes and sizes to fit various vehicle designs.
    - Adapting the battery for extreme weather conditions.

    - **Creative suggestions:**
    - Design a universal battery module that can be adapted for different vehicles by changing its configuration or adding supplementary modules.
    - Develop a battery with adaptive thermal management to optimize performance in various climate conditions.
    - Create a battery that can be easily reconfigured for use in stationary energy storage, powering homes or businesses when not in use in a vehicle.

    ### 5. Transform:
    - **Intuitive insights:**
    - Enhancing battery energy density and longevity.
    - Improving charging speed and efficiency.
    - Transforming the battery’s form factor for better integration into vehicle designs.

    - **Creative suggestions:**
    - Innovate a flexible battery that can be molded into different shapes, allowing for more creative vehicle designs.
    - Develop a self-healing battery that can automatically repair minor damages, extending its lifespan.
    - Implement wireless charging capabilities to simplify the charging process and reduce wear on physical connectors.

    ### 6. Explore Other Uses:
    - **Intuitive insights:**
    - Using EV batteries for grid storage when they are no longer efficient for vehicle use.
    - Employing old EV batteries in renewable energy projects.
    - Utilizing EV batteries in portable power solutions.

    - **Creative suggestions:**
    - Repurpose EV batteries for use in marine applications, such as electric boats or underwater drones.
    - Explore the use of EV batteries in aerospace applications, providing power for electric planes or satellites.
    - Develop a compact version of the EV battery for use in portable emergency power systems or for camping and outdoor activities.

    ### Summary:

    - **Combine:**
    - Intuitive insights: Integrating with renewable energy, smart technology, and home energy systems.
    - Creative suggestions: Hybrid systems, AI-driven maintenance, modular designs.

    - **Reverse:**
    - Intuitive insights: Recycling, second-life applications, reversible charging.
    - Creative suggestions: Easy disassembly, leasing model, dual-function batteries.

    - **Eliminate:**
    - Intuitive insights: Removing harmful materials, simplifying design, finding alternative resources.
    - Creative suggestions: Bio-based materials, streamlined management systems, 3D printing.

    - **Adapt:**
    - Intuitive insights: Adjusting capacity, customizing shapes, adapting for weather conditions.
    - Creative suggestions: Universal modules, adaptive thermal management, reconfigurable for stationary storage.

    - **Transform:**
    - Intuitive insights: Improving density, charging speed, and form factor.
    - Creative suggestions: Flexible batteries, self-healing, wireless charging.

    - **Explore Other Uses:**
    - Intuitive insights: Grid storage, renewable energy projects, portable power.
    - Creative suggestions: Marine and aerospace applications, compact emergency power systems."


    ##Your input : {idea}
    Your output:

    Give output in markdown format.
    """
    response = llm.invoke(prompt)
    return response.content

#Function to do Attribute Analysis
def attribute_analysis(llm, idea):
    prompt = f"""
        ## Instruction ##
        Conduct a comprehensive and innovative attribute analysis for the following idea: {idea}

        Follow these detailed steps:
        1. Identify 4 main categories of attributes relevant to the idea.
        2. For each category, create a table with 4 attributes and 4 possible values for each attribute.
        3. After the tables, create 5 hypothetical systems or products based on combinations of these attributes, ranging from realistic to highly innovative.
        4. For each hypothetical system, create a table showing the selected values for 8 attributes (2 from each category).
        5. After each system's table, provide 1-2 brief points about its potential usefulness or applications.

        ## Some more detailed instructions to be noted##
        6. Create 5-6 unique combinations of attribute values from different categories to generate innovative ideas for the product, service, or technology. Start with realistic combinations that are more common in the current market landscape and gradually progress towards more innovative and futuristic combinations.
        7. For each combination, provide a brief but informative analysis of its usefulness and potential applications. Highlight the specific benefits, target audience, or situations where the combination would be most effective.
        8. When creating combinations, ensure that they are diverse and cover a wide range of attributes from different categories. Each combination should demonstrate creative thinking and cater to various user needs and scenarios.
        9. As you progress towards more innovative combinations, consider incorporating advanced technologies, emerging trends, or unconventional approaches. This showcases an understanding of the evolving market landscape and a forward-thinking mindset.
        10. Ensure that each combination is coherent and has a logical flow. The selected attribute values should complement each other, creating a cohesive and practical solution.
        11. Consider the impact of your combinations on various stakeholders, including end-users, businesses, society, and the environment. Incorporate attributes that address sustainability, inclusivity, and ethical considerations when relevant.
        12. Present your combinations in a clear and engaging writing style that effectively communicates your ideas and insights. Provide sufficient context and explanations to make the response informative and valuable to the reader.
        13. Conclude your Attribute Analysis with a summary of the key insights gained and the potential impact of the innovative combinations on the chosen product, service, or technology.



        Use the following format for your analysis:

        Category 1: [Name]

        | Attribute | Values |
        |-----------|--------|
        | [Attr 1]  | 1. [Value 1], 2. [Value 2], 3. [Value 3], 4. [Value 4] |
        | [Attr 2]  | 1. [Value 1], 2. [Value 2], 3. [Value 3], 4. [Value 4] |
        | [Attr 3]  | 1. [Value 1], 2. [Value 2], 3. [Value 3], 4. [Value 4] |
        | [Attr 4]  | 1. [Value 1], 2. [Value 2], 3. [Value 3], 4. [Value 4] |

        [Repeat this table format for all 4 categories]

        Now, predict 5 systems based on combinations of these attributes, starting from realistic and progressing to more complex and innovative:

        1. "[System Name 1]"

        | Category | Attribute | Selected Value |
        |----------|-----------|----------------|
        | [Category 1] | [Attribute] | [Selected Value] |
        | [Category 1] | [Attribute] | [Selected Value] |
        | [Category 2] | [Attribute] | [Selected Value] |
        | [Category 2] | [Attribute] | [Selected Value] |
        | [Category 3] | [Attribute] | [Selected Value] |
        | [Category 3] | [Attribute] | [Selected Value] |
        | [Category 4] | [Attribute] | [Selected Value] |
        | [Category 4] | [Attribute] | [Selected Value] |

        Usefulness:
        - [Point 1]
        - [Point 2]

        [Repeat this format for all 5 systems, increasing in complexity and innovation]

        Here's an example analysis for smartphones to guide your response:

        Thanks for your input. Based on your input: Smartphones, here are the 4 categories of attributes we came up with!

        Let's look at them one by one.

        Category 1: Display

        | Attribute | Values |
        |-----------|--------|
        | Type | 1. LCD, 2. OLED, 3. MicroLED, 4. Foldable |
        | Size | 1. Compact (5-5.9"), 2. Standard (6-6.9"), 3. Large (7"+), 4. Variable |
        | Refresh Rate | 1. 60Hz, 2. 90Hz, 3. 120Hz, 4. Adaptive (1-120Hz) |
        | Resolution | 1. HD+, 2. FHD+, 3. QHD+, 4. 4K |

        Category 2: Performance

        | Attribute | Values |
        |-----------|--------|
        | Processor | 1. Entry-level, 2. Mid-range, 3. Flagship, 4. AI-optimized |
        | RAM | 1. 4GB, 2. 8GB, 3. 12GB, 4. 16GB+ |
        | Storage | 1. 64GB, 2. 128GB, 3. 256GB, 4. 512GB+ |
        | Battery | 1. 3000mAh, 2. 4000mAh, 3. 5000mAh, 4. 6000mAh+ |

        Category 3: Camera System

        | Attribute | Values |
        |-----------|--------|
        | Main Sensor | 1. 12MP, 2. 48MP, 3. 64MP, 4. 108MP+ |
        | Zoom | 1. Digital, 2. 2x optical, 3. 5x optical, 4. 10x optical |
        | Video | 1. 1080p60fps, 2. 4K30fps, 3. 4K60fps, 4. 8K30fps |
        | Special Features | 1. Portrait mode, 2. Night mode, 3. AI enhancement, 4. Computational photography |

        Category 4: Connectivity

        | Attribute | Values |
        |-----------|--------|
        | Network | 1. 4G, 2. 5G, 3. 5G mmWave, 4. Satellite |
        | Wi-Fi | 1. Wi-Fi 5, 2. Wi-Fi 6, 3. Wi-Fi 6E, 4. Wi-Fi 7 |
        | Charging | 1. Wired, 2. Fast wired, 3. Wireless, 4. Reverse wireless |
        | Biometrics | 1. Fingerprint, 2. Face unlock, 3. In-display fingerprint, 4. 3D face unlock |


        Now, let's predict some smartphone models based on combinations of these attributes, starting from realistic and progressing to more innovative:

        1. "EcoBalance"

        | Category | Attribute | Selected Value |
        |----------|-----------|----------------|
        | Display | Type | OLED |
        | Display | Size | Compact (5-5.9") |
        | Performance | Processor | Mid-range |
        | Performance | Battery | 4000mAh |
        | Camera System | Main Sensor | 48MP |
        | Camera System | Special Features | AI enhancement |
        | Connectivity | Network | 5G |
        | Connectivity | Charging | Fast wired |

        Usefulness:
        - Energy-efficient design for environmentally conscious consumers
        - Compact size and AI-enhanced camera for on-the-go photography enthusiasts

        This combination addresses the growing demand for sustainable technology while maintaining essential features for daily use. The compact size and efficient battery make it ideal for users who prioritize portability and longer battery life. The AI-enhanced camera caters to the increasing interest in mobile photography without the need for professional-grade equipment.

        2. "WorkFlow Pro"

        | Category | Attribute | Selected Value |
        |----------|-----------|----------------|
        | Display | Type | OLED |
        | Display | Refresh Rate | 120Hz |
        | Performance | Processor | Flagship |
        | Performance | RAM | 12GB |
        | Camera System | Zoom | 5x optical |
        | Camera System | Video | 4K60fps |
        | Connectivity | Wi-Fi | Wi-Fi 6E |
        | Connectivity | Biometrics | In-display fingerprint |

        Usefulness:
        - Enhanced productivity features for business professionals and power users
        - Seamless multitasking and secure access for handling sensitive information

        This combination targets professionals who require high performance and security in their mobile devices. The high refresh rate and powerful processor enable smooth multitasking, while advanced biometrics ensure data protection. The versatile camera system supports various business needs, from document scanning to high-quality video conferencing.

        3. "FoldFlex Ultra"

        | Category | Attribute | Selected Value |
        |----------|-----------|----------------|
        | Display | Type | Foldable |
        | Display | Size | Variable |
        | Performance | Processor | AI-optimized |
        | Performance | Storage | 512GB+ |
        | Camera System | Main Sensor | 108MP+ |
        | Camera System | Special Features | Computational photography |
        | Connectivity | Network | 5G mmWave |
        | Connectivity | Charging | Wireless |

        Usefulness:
        - Adaptable form factor for diverse use cases, from pocket-sized to tablet mode
        - Advanced AI and camera capabilities for creative professionals and tech enthusiasts

        This innovative combination pushes the boundaries of smartphone versatility. The foldable display and AI-optimized processor cater to users who demand flexibility in their devices, from compact everyday use to expanded productivity or entertainment modes. The advanced camera system with computational photography appeals to content creators and photography enthusiasts.

        4. "HealthGuardian X"

        | Category | Attribute | Selected Value |
        |----------|-----------|----------------|
        | Display | Type | MicroLED |
        | Display | Refresh Rate | Adaptive (1-120Hz) |
        | Performance | Processor | AI-optimized |
        | Performance | Battery | 6000mAh+ |
        | Camera System | Special Features | AI enhancement |
        | Camera System | Main Sensor | 64MP |
        | Connectivity | Network | Satellite |
        | Connectivity | Biometrics | 3D face unlock |

        Usefulness:
        - Comprehensive health monitoring and emergency communication capabilities
        - Long-lasting battery and adaptive display for outdoor and extreme conditions

        This forward-thinking combination focuses on health and safety. The AI-optimized processor enables advanced health monitoring algorithms, while satellite connectivity ensures communication in remote areas. The long-lasting battery and adaptive display make it suitable for outdoor enthusiasts and individuals with health concerns who require constant monitoring.

        5. "NeuroLink Infinity"

        | Category | Attribute | Selected Value |
        |----------|-----------|----------------|
        | Display | Type | Holographic |
        | Display | Resolution | 4K |
        | Performance | Processor | Quantum |
        | Performance | RAM | 16GB+ |
        | Camera System | Special Features | Brain-computer interface |
        | Camera System | Video | 8K30fps |
        | Connectivity | Network | 6G |
        | Connectivity | Biometrics | Thought patterns |

        Usefulness:
        - Revolutionary human-computer interaction through neural interface
        - Immersive holographic display for advanced augmented and virtual reality applications

        This highly innovative combination represents a potential future of mobile technology. The brain-computer interface and thought pattern biometrics offer unprecedented control and security. The holographic display and quantum processor enable complex AR/VR applications, potentially transforming fields like education, healthcare, and entertainment. While futuristic, this combination raises important ethical considerations regarding privacy and the integration of technology with human cognition.

        Summary:
        These combinations demonstrate a progression from current market-ready solutions to highly innovative concepts. They address various user needs, from sustainability and productivity to health monitoring and futuristic interactions. The analysis highlights the potential impact on different user groups and industries, while also considering technological trends and ethical implications. As smartphone technology continues to evolve, these combinations suggest potential directions for innovation, emphasizing the importance of balancing functionality, user experience, and societal impact.

        Use this example as a guide to create a similar attribute analysis for: {idea}

        ## End Instruction ##
        """

    response = llm.invoke(prompt)
    return response.content

def morphological_analysis(llm, idea):
    prompt = f"""
        ## Instruction ##
        Conduct a comprehensive and innovative morphological analysis for the following idea: {idea}

        Follow these detailed steps:
        1. Identify 4 main categories of attributes relevant to the idea.
        2. For each category, create a table with 4 attributes and 4 possible values for each attribute.
        3. After the tables, create 5 hypothetical systems or products by replacing the original attributes with elements from adjacent or complementary domains. These systems should range from realistic to highly innovative.
        4. For each hypothetical system, create a table showing 8 replaced elements (2 from each category) and their original counterparts.
        5. After each system's table, provide 1-2 brief points about its potential usefulness or applications.

        Use the following format for your analysis:

        Category 1: [Name]

        | Attribute | Values |
        |-----------|--------|
        | [Attr 1]  | 1. [Value 1], 2. [Value 2], 3. [Value 3], 4. [Value 4] |
        | [Attr 2]  | 1. [Value 1], 2. [Value 2], 3. [Value 3], 4. [Value 4] |
        | [Attr 3]  | 1. [Value 1], 2. [Value 2], 3. [Value 3], 4. [Value 4] |
        | [Attr 4]  | 1. [Value 1], 2. [Value 2], 3. [Value 3], 4. [Value 4] |

        [Repeat this table format for all 4 categories]

        Now, create 5 innovative systems by replacing attributes with elements from adjacent or complementary domains:

        1. "[System Name 1]"

        | Original Category | Original Attribute | Replaced Element | Adjacent/Complementary Domain |
        |--------------------|---------------------|-------------------|--------------------------------|
        | [Category 1] | [Attribute] | [Replaced Element] | [Domain] |
        | [Category 1] | [Attribute] | [Replaced Element] | [Domain] |
        | [Category 2] | [Attribute] | [Replaced Element] | [Domain] |
        | [Category 2] | [Attribute] | [Replaced Element] | [Domain] |
        | [Category 3] | [Attribute] | [Replaced Element] | [Domain] |
        | [Category 3] | [Attribute] | [Replaced Element] | [Domain] |
        | [Category 4] | [Attribute] | [Replaced Element] | [Domain] |
        | [Category 4] | [Attribute] | [Replaced Element] | [Domain] |

        Usefulness:
        - [Point 1]
        - [Point 2]

        [Repeat this format for all 5 systems, increasing in complexity and innovation]

        Here's an example analysis for smartphones to guide your response:

        Category 1: Display

        | Attribute | Values |
        |-----------|--------|
        | Type | 1. LCD, 2. OLED, 3. MicroLED, 4. Foldable |
        | Size | 1. Compact (5-5.9"), 2. Standard (6-6.9"), 3. Large (7"+), 4. Variable |
        | Refresh Rate | 1. 60Hz, 2. 90Hz, 3. 120Hz, 4. Adaptive (1-120Hz) |
        | Resolution | 1. HD+, 2. FHD+, 3. QHD+, 4. 4K |


        Category 2: Performance

        | Attribute | Values |
        |-----------|--------|
        | Processor | 1. Entry-level, 2. Mid-range, 3. Flagship, 4. AI-optimized |
        | RAM | 1. 4GB, 2. 8GB, 3. 12GB, 4. 16GB+ |
        | Storage | 1. 64GB, 2. 128GB, 3. 256GB, 4. 512GB+ |
        | Battery | 1. 3000mAh, 2. 4000mAh, 3. 5000mAh, 4. 6000mAh+ |

        Category 3: Camera System

        | Attribute | Values |
        |-----------|--------|
        | Main Sensor | 1. 12MP, 2. 48MP, 3. 64MP, 4. 108MP+ |
        | Zoom | 1. Digital, 2. 2x optical, 3. 5x optical, 4. 10x optical |
        | Video | 1. 1080p60fps, 2. 4K30fps, 3. 4K60fps, 4. 8K30fps |
        | Special Features | 1. Portrait mode, 2. Night mode, 3. AI enhancement, 4. Computational photography |

        Category 4: Connectivity

        | Attribute | Values |
        |-----------|--------|
        | Network | 1. 4G, 2. 5G, 3. 5G mmWave, 4. Satellite |
        | Wi-Fi | 1. Wi-Fi 5, 2. Wi-Fi 6, 3. Wi-Fi 6E, 4. Wi-Fi 7 |
        | Charging | 1. Wired, 2. Fast wired, 3. Wireless, 4. Reverse wireless |
        | Biometrics | 1. Fingerprint, 2. Face unlock, 3. In-display fingerprint, 4. 3D face unlock |


        Now, let's create innovative systems by replacing attributes with elements from adjacent or complementary domains:

        1. "NeuroPulse Communicator"

        | Original Category | Original Attribute | Replaced Element | Adjacent/Complementary Domain |
        |--------------------|---------------------|-------------------|--------------------------------|
        | Display | Type | Neural projection | Neuroscience |
        | Display | Size | Adjustable mental canvas | Virtual Reality |
        | Performance | Processor | Quantum neural network | Quantum Computing |
        | Performance | RAM | Biological memory augmentation | Biotechnology |
        | Camera System | Main Sensor | Synesthetic capture | Cognitive Science |
        | Camera System | Special Features | Emotion recognition | Psychology |
        | Connectivity | Network | Thought-based mesh network | Telepathy Research |
        | Connectivity | Biometrics | DNA signature | Genetics |

        Usefulness:
        - Revolutionary communication device that translates thoughts and emotions into shareable experiences
        - Potential applications in mental health, education, and interpersonal understanding

        2. "EcoSphere Nexus"

        | Original Category | Original Attribute | Replaced Element | Adjacent/Complementary Domain |
        |--------------------|---------------------|-------------------|--------------------------------|
        | Display | Type | Bioluminescent screen | Marine Biology |
        | Display | Refresh Rate | Circadian rhythm sync | Chronobiology |
        | Performance | Battery | Photosynthetic power cells | Botany |
        | Performance | Storage | Organic molecular storage | Biochemistry |
        | Camera System | Zoom | Compound eye array | Entomology |
        | Camera System | Video | Pheromone-based encoding | Chemical Ecology |
        | Connectivity | Charging | Ambient energy harvesting | Environmental Science |
        | Connectivity | Wi-Fi | Mycorrhizal network interface | Mycology |

        Usefulness:
        - Eco-friendly device that integrates seamlessly with natural ecosystems
        - Applications in environmental monitoring, sustainable living, and biome-human interfaces

        3. "Quantum Entanglement Communicator"

        | Original Category | Original Attribute | Replaced Element | Adjacent/Complementary Domain |
        |--------------------|---------------------|-------------------|--------------------------------|
        | Display | Type | Holographic quantum field | Quantum Optics |
        | Display | Resolution | Planck-scale pixelation | Theoretical Physics |
        | Performance | Processor | Quantum entanglement computer | Quantum Information Theory |
        | Performance | RAM | Infinite quantum superposition | Multiverse Theory |
        | Camera System | Main Sensor | Cosmic ray detector | Particle Physics |
        | Camera System | Special Features | Parallel universe imaging | String Theory |
        | Connectivity | Network | Instantaneous quantum tunneling | Quantum Teleportation |
        | Connectivity | Biometrics | Quantum state authentication | Quantum Cryptography |

        Usefulness:
        - Revolutionizes communication by enabling instantaneous, secure connections across vast distances
        - Potential applications in deep space exploration, alternate reality research, and unhackable communications

        4. "Temporal Nexus Device"

        | Original Category | Original Attribute | Replaced Element | Adjacent/Complementary Domain |
        |--------------------|---------------------|-------------------|--------------------------------|
        | Display | Type | Temporal projection screen | Theoretical Physics (Time) |
        | Display | Refresh Rate | Time dilation synchronization | Relativity Theory |
        | Performance | Processor | Causality manipulation engine | Philosophy of Time |
        | Performance | Storage | Akashic field access | Metaphysics |
        | Camera System | Zoom | Chronological focus | Archaeology |
        | Camera System | Video | Historical event reconstruction | Historiography |
        | Connectivity | Network | Temporal rift network | Science Fiction Concepts |
        | Connectivity | Charging | Entropy reversal power | Thermodynamics |

        Usefulness:
        - Allows users to observe and interact with different points in time
        - Applications in historical research, predictive modeling, and personal life reflection

        5. "Consciousness Integration Matrix"

        | Original Category | Original Attribute | Replaced Element | Adjacent/Complementary Domain |
        |--------------------|---------------------|-------------------|--------------------------------|
        | Display | Type | Mind-matter interface | Consciousness Studies |
        | Display | Size | Infinite mental space | Cognitive Architecture |
        | Performance | Processor | Collective consciousness hub | Social Psychology |
        | Performance | RAM | Akashic record access | Mysticism |
        | Camera System | Main Sensor | Reality perception filter | Phenomenology |
        | Camera System | Special Features | Qualia manipulation | Philosophy of Mind |
        | Connectivity | Network | Noosphere integration | Teilhard de Chardin's Theory |
        | Connectivity | Biometrics | Soul signature recognition | Spirituality |

        Usefulness:
        - Transcends individual consciousness limitations by connecting to a global consciousness network
        - Potential to revolutionize human understanding, problem-solving, and spiritual experiences

        Ensure that the replaced elements come from truly adjacent or complementary domains, leading to creative and unexpected combinations. The systems should progress from relatively realistic to highly innovative and potentially disruptive concepts.

        Conclude your Morphological Analysis with a summary of the key insights gained and the potential impact of these innovative combinations on the chosen product, service, or technology.

        ## End Instruction ##
        """

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


def market_analysis(llm, idea, market_data):
    prompt = f"""
    ## Instruction ##
    Analyze and summarize the provided market data according to the following components, using inline citations where possible:

    1. Market size: Estimate the total market size based on the given data.
    2. Sales analysis: Examine sales trends, patterns, and key performance indicators.
    3. Geographic location: Identify key regions or areas of interest in the market.
    4. Market segmentation: Divide the market into distinct groups based on shared characteristics.
    5. Demographic description: Describe the key demographic factors of the target market.
    6. Analysis of market demand: Evaluate current and projected demand for products or services.
    7. Competition in the market: Identify major competitors and assess their market positions.
    8. Consumer insights & requirements: Highlight key consumer needs, preferences, and behaviors.

    Provide a concise summary for each component based on the information in the market_data. If any component lacks sufficient information, mention that the data is limited or unavailable for that aspect.

    Use inline citations in the format (Source "URL") where X is the URL of the source document. For example, (Source www.xyz.com) for information for the mentioned point.

    This is the domain whose market has to be searched: {idea}

    Market Data:
    {market_data}

    Your analysis should be clear, data-driven, and actionable for business decision-making. Ensure to cite the sources for key information and statistics where possible.
    """

    response = llm.invoke(prompt)
    return response.content


def convo():
    st.title("Kreat Conversation")
    st.header("Converse with Kreat")


     # Sidebar navigation
    st.sidebar.title("Function Navigator")
    function_names = [
        "Extract Problem Information✅",
        "Generate Title✅",
        "Check Title✅",
        "Update Title✅",
        "Generate Abstract✅",
        "Update Abstract✅",
        "Assess Problem Type✅",
        "Explain Problem Type Assessment✅",
        "Visualize Sliders✅",
        "Generate Assumptions✅",
        "Access Data Sources",
        "Summarize Key Findings",
        "Update Problem Description✅",
        "Suggest PBD model✅",
        "PBD models✅",
        "Analyze Problem Breadth and Depth✅",
        "Update Breadth and Depth✅",
        "Generate Future Scenarios",
        "Create Problem Landscape✅",
        "Create Function Map✅",
        "CREATE Model for ideas✅",
        "Attribute Analysis✅",
        "Morphological Analysis✅",
        "Market analysis✅",
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

    if choice =="Extract Problem Information✅":
        goal = st.text_input("Goal")
        if st.button("Run"):
            with st.spinner("Extracting Information...."):
                extracted_information = problem_extraction(llm,goal)
            st.code(extracted_information)
            response = parse_problem_extraction(extracted_information)
            st.write(response)

    elif choice == "Generate Title✅":
        extracted_information = st.text_area("Input the extracted Information(user won't need to enter): ")
        if st.button("Run"):
            with st.spinner("Generating Title..."):
                result = generate_title(llm,extracted_information)
            st.code(result)
            st.write(parse_title_generation(result))

    elif choice == "Update Title✅":
        current_title = st.text_input("Current Title(User won't need to enter)")
        feedback = st.text_input("Feedback")
        if st.button("Run"):
            result = update_title(llm,current_title, feedback)
            st.write(parse_title_update(result))

    elif choice == "Generate Abstract✅":
        title = st.text_input("Title(User won't need to enter)")
        extracted_information = st.text_area("Enter the extracted information(User won't need to enter): ")
        if st.button("Run"):
            result = generate_abstract(llm,title,extracted_information)
            st.write(result)
            st.write(parse_abstract_generation(result))

    elif choice == "Update Abstract✅":
        current_abstract = st.text_input("Current Abstract")
        feedback = st.text_input("Feedback")
        if st.button("Run"):
            result = update_abstract(llm,current_abstract, feedback)
            st.write(result)
            st.write(parse_abstract_update(result))

    elif choice == "Assess Problem Type✅":
        extracted_problems = st.text_area("Enter the extracted problems(User would not have to enter):")
        if st.button("Run"):
            with st.spinner("Assessing Problem type..."):
                problem_classification = assess_problem(llm,extracted_problems)
            st.code(problem_classification)
            st.write(parse_problem_assessment(problem_classification))

    elif choice == "Explain Problem Type Assessment✅":
        extracted_problems = st.text_area("Enter the extracted problems(User would not have to enter):")
        if st.button("Run"):
            with st.spinner("Assessing Problem type..."):
                problem_classification = assess_problem(llm,extracted_problems)
            with st.spinner("Reasoning Problem Classifications...."):
                explained_problem_classification = explain_problem_classification(llm,extracted_problems,problem_classification)
            st.code(explained_problem_classification)

    elif choice == "Visualize Sliders✅":
        complexity = st.slider("How would you rate the complexity of your problem on a scale of 1-10?", 1, 10, 1)
        predictability = st.slider("How would you rate the predictability of your problem on a scale of 1-10?", 1, 10, 1)
        extracted_problems = st.text_area("Enter the extracted problems(User would not have to enter):")
        if st.button("Run:"):
            with st.spinner("Analyzing your problem now..."):
                problem_classification = user_enhanced_problem_classification(llm,extracted_problems,complexity,predictability)
            st.code(problem_classification)

    elif choice == "Generate Assumptions✅":
        extracted_problems = st.text_area("Enter the extracted problems(User would not have to enter):")
        if st.button("Run"):
            with st.spinner("Generating Assumptions..."):
                assumptions = generate_assumptions(llm,extracted_problems)
            st.code(assumptions)
            st.write(parse_assumptions(assumptions))


    elif choice == "Access Data Sources":
        st.write("Prompt Under Development")
        
    elif choice == "Summarize Key Findings":
        pass

    elif choice == "Update Problem Description✅":
        extracted_problems = st.text_area("Enter extracted problems(user won't have to add):")
        if st.button("Run"):
            with st.spinner("Generating and updating description..."):
                description = generate_description(llm,extracted_problems)
            st.write(description)

    elif choice == "Suggest PBD model✅":
        problem = st.text_input("Enter a problem(User won't have to enter):")
        if st.button("Run"):
            with st.spinner("Suggesting think model for you:"):
                suggestion_output = suggest_pdb_model(llm,problem)
                #st.write(suggestion_output)
                suggestion_parsed = parse_pbd_suggestion(suggestion_output)
                st.write(suggestion_parsed)
    
    elif choice == "PBD models✅":
        problem = st.text_input("Enter your problem(Users won't have to enter):")
        def model_5ws_and_h():
            st.header("5Ws and H Model")
            st.write(analyze_with_5w1h(llm,problem))

        def model_5ps():
            st.header("5Ps Model")
            st.write(analyze_with_5ps(llm,problem))

        def model_5ms():
            st.header("5Ms Model")
            st.write(analyze_with_5ms(llm,problem))

        def model_5es():
            st.header("5Es Model")
            st.write(analyze_with_5es(llm,problem))

        def model_4ps():
            st.header("4Ps Model")
            st.write(analyze_with_4ps(llm,problem))
        # Define the models
        models = {
            "5Ws and H": model_5ws_and_h,
            "5Ps": model_5ps,
            "5Ms": model_5ms,
            "5Es": model_5es,
            "4Ps": model_4ps
        }

        # Create the radio button in the main area
        model_selection = st.radio("Select a Model", list(models.keys()))

        # Add some space
        st.write("")

        # Display the selected model
        models[model_selection]()

    elif choice == "Check Title✅":
        title = st.text_input("Enter a title:")
        if st.button("Run"):
            evaluation = check_title(llm,title)
            st.write(parse_title_check(evaluation))
        
    elif choice == "Analyze Problem Breadth and Depth✅":
        extracted_problems = st.text_area("Enter extracted problems(user won't have to add):")
        if st.button("Run"):
            with st.spinner("Generating problem breadth and depth..."):
                problem_breadth_depth = generate_breadth_and_depth(llm,extracted_problems)
            st.markdown(problem_breadth_depth)

    elif choice == "Update Breadth and Depth✅":
        current_breadth_depth = st.text_area("Enter current problem breadth and depth(user wont have to add):")
        feedback = st.text_area("Enter input:")
        if st.button("Run"):
            with st.spinner("Updating problem breadth and depth..."):
                updated_depth_breadth = update_depth_breadth(llm,current_breadth_depth,feedback)
            st.markdown(updated_depth_breadth)

    elif choice == "Generate Future Scenarios":
        st.write("Prompt Under Development")

    elif choice == "Create Problem Landscape✅":
        extracted_information = st.text_input("Enter extracted problems(user won't have to add):")
        if st.button("Run"):
            with st.spinner("Creating Problem Landscape...."):
                functionmap = problem_landscape(llm,extracted_information)
            #st.write(functionmap)
            table,parsed_data = parse_problem_landscape_output(functionmap)
            st.table(table)
           # Store the parsed data in session state
            st.session_state.parsed_data = parsed_data

            # Check if parsed data is in session state to display multiselect
            if 'parsed_data' in st.session_state:
                # Convert the dictionary to a list of tuples (key, value) for multiselect options
                options = [(key, value) for key, value in st.session_state.parsed_data.items()]

                # Multiselect widget for selecting multiple values
                selected_keys = st.multiselect("Select options:", options, format_func=lambda x: f"{x[0]}: {x[1]}")

                # Button to print the list of selected values
                if st.button("Print Selected Values"):
                    selected_values = [value for key, value in selected_keys]
                    st.write("Selected Values:", selected_values)

    elif choice == "Create Function Map✅":
        components = st.text_input("Enter components:")
        if st.button("Run"):
            with st.spinner("Creating Useful Function map..."):
                st.markdown("### Function Map with Useful Functions")
                st.write(identify_useful_function_map(llm,components))
            with st.spinner("Creating Harmful Function map..."):
                st.markdown("### Function Map with Harmful Functions")
                st.write(identify_harmful_function_map(llm,components))

    elif choice == "CREATE Model for ideas✅":
        idea = st.text_input("Enter any idea:")
        if st.button("Run"):
            with st.spinner("Let's CREATE your idea..."):
                st.markdown(create_same_domain_prompt(llm,idea))
            with st.spinner("Let's CREATE your idea from adjacent domain..."):
                st.markdown(create_adjacent_domain_prompt(llm,idea))


    elif choice == "Attribute Analysis✅":
        idea = st.text_input("Enter any idea: ")
        if st.button("Run"):
            with st.spinner("Performing Attribute Analysis...."):
                st.markdown(attribute_analysis(llm,idea))

    elif choice == "Morphological Analysis✅":
        idea = st.text_input("Enter any idea: ")
        if st.button("Run"):
            with st.spinner("Performing Morphological Analysis...."):
                st.markdown(morphological_analysis(llm,idea))

    elif choice == "Market analysis✅":
        idea = st.text_input("Enter an idea(Users won't have to enter):")
        if st.button("Run"):
            with st.spinner("Extracting Market Data: "):
                prompt = f"""
                Here is an idea: {idea}. 
                Give me market trends, opportunities, and competition for this idea for the current market for this idea.
                """
                data = search_and_extract(prompt)
                market_data = ""
                for d in data:
                    url = d.url
                    title = d.title
                    date = d.published_date
                    author = d.author
                    text = d.text
                    split_market_data = market_data.split()
                    if len(split_market_data)<14000:
                        market_data += f"""\n Market Content 
                        \nTitle:{title} 
                        \nDate: {date}
                        \nAuthor: {author}
                        \nText: {text}
                        \nURL: {url}
                        """
                #st.write(market_data)        
            with st.spinner("Generating Market Analysis..."):
                st.write(market_analysis(llm,idea,market_data))      

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



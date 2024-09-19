import streamlit as st
from langchain_core.prompts import PromptTemplate 
from langchain_openai import OpenAI
#from dotenv import load_dotenv, find_dotenv
import os
from crewai import Crew

from tasks import MeetingPreparationTasks
from agents import MeetingPreparationAgents
from dependencies import check_password

os.environ["GROQ_API_KEY"] = st.secrets.APIKEY.GROQ_API_KEY
os.environ['SERPER_API_KEY'] = st.secrets.APIKEY.SERPER_API_KEY
os.environ['BROWSERLESS_API_KEY'] = st.secrets.APIKEY.BROWSERLESS_API_KEY
os.environ['MODEL']= st.secrets.APIKEY.MODEL
os.environ['EXA_API_KEY'] = st.secrets.APIKEY.EXA_API_KEY
os.environ['OPENAI_API_KEY'] = st.secrets.APIKEY.OPENAI_API_KEY


openai_api_key = os.environ["OPENAI_API_KEY"]

tasks = MeetingPreparationTasks()
agents = MeetingPreparationAgents()

####################
#### STREAMLIT #####
####################

st.set_page_config(
    page_title="Agents Assistant",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://genaiexpertise.com/contact/",    
        "Report a bug": "https://github.com/genaiexpertise/RAGApps/issues",
        "About": "https://genaiexpertise.com",
    },
)
st.logo(
    'assets/logo.png',
    link="https://genaiexpertise.com",
    icon_image="assets/icon.png",
)


if not check_password():
    st.stop()

st.header("Meetings Assistant")

col1,col2,col3 = st.columns(3)

with col1:
    st.subheader("Participants")
    participants = st.text_area("Enter the participants separated by commas")
    # participants = participants.split(',')
    # participants = [participant.strip() for participant in participants]

with col2:
    st.subheader("Meeting Context")
    context = st.text_area("Enter the context of the meeting")

with col3:
    st.subheader("Meeting Objective")
    objective = st.text_area("Enter the objective of the meeting")


# assign agents
researcher_agent = agents.research_agent()
industry_analyst_agent = agents.industry_analysis_agent()
meeting_strategy_agent = agents.meeting_strategy_agent()
summary_and_briefing_agent = agents.summary_and_briefing_agent()


# assign tasks
# research = tasks.research_task(researcher_agent, participants, context)
# industry_analysis = tasks.industry_analysis_task(industry_analyst_agent, participants, context)
# meeting_strategy = tasks.meeting_strategy_task(meeting_strategy_agent, context, objective)
# summary_and_briefing = tasks.summary_and_briefing_task(summary_and_briefing_agent, context, objective)

# meeting_strategy.context = [research, industry_analysis]
# summary_and_briefing.context = [research, industry_analysis, meeting_strategy]



# run the crew
if st.button("Run Crew"):
    with st.spinner("Thinking..."):
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Research Task")
            research = tasks.research_task(researcher_agent, participants, context)
            st.write(research.description)
        with col2:
            st.subheader("Industry Analysis Task")
            industry_analysis = tasks.industry_analysis_task(industry_analyst_agent, participants, context)
            st.write(industry_analysis.description)

        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Meeting Strategy Task")
            meeting_strategy = tasks.meeting_strategy_task(meeting_strategy_agent, context, objective)
            st.write(meeting_strategy.description)
        with col2:
            st.subheader("Summary and Briefing Task")
            summary_and_briefing = tasks.summary_and_briefing_task(summary_and_briefing_agent, context, objective)
            st.write(summary_and_briefing.description)

        meeting_strategy.context = [research, industry_analysis]
        summary_and_briefing.context = [research, industry_analysis, meeting_strategy]

        # create the crew
        crew = Crew(
            agents=[
                researcher_agent,
                industry_analyst_agent,
                meeting_strategy_agent,
                summary_and_briefing_agent
            ],
            tasks=[
                research,
                industry_analysis,
                meeting_strategy,
                summary_and_briefing
            ]
        )


        result = crew.kickoff()
        if result:
            st.success("Crew has been successfully executed!")
            st.markdown(f"Download the research report [here](assets/research_report.md)")
            st.markdown(f"Download the industry analysis report [here](assets/industry_analysis_report.md)")
            st.markdown(f"Download the meeting strategy report [here](assets/meeting_strategy_report.md)")
            st.markdown(f"Download the summary and briefing report [here](assets/summary_and_briefing_report.md)")
            st.markdown(f"Below are the results of the crew execution:")
            st.write(result)




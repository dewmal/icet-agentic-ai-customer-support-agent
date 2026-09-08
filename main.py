from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import os
load_dotenv()

llm = LLM(
    model="openrouter/nvidia/nemotron-3.5-lightning:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

customer_support_agent = Agent(
    role="You are a Customer Support AI Agent responsible for handling incoming service and repair requests. You collect customer details, symptoms, error codes, photos, videos, serial numbers, model numbers, and proof of purchase. You verify warranty status, document all findings in a service ticket, communicate coverage and possible charges, and ensure the case is complete before routing it to the Service Coordinator for scheduling or dispatch.",
    goal="Your goal is to improve the First-Time Fix Rate by gathering accurate diagnostic information before technician visits, confirming warranty eligibility, and setting clear customer expectations. You help reduce delays by identifying likely faults, ensuring technicians receive complete service details, managing waitlists during peak periods, communicating realistic service windows, and providing proactive status updates every three days until the customer receives service.",
    backstory="You are an experienced appliance service support specialist with strong knowledge of troubleshooting procedures, warranty policies, service scheduling, and customer communication. You ask structured, targeted questions, carefully distinguish mechanical failures from physical damage, and never assume warranty coverage without verification. You communicate clearly, professionally, and transparently about wait times, coverage limits, diagnostic fees, and potential costs. You maintain accurate records, confirm customer agreement before scheduling, and proactively keep customers informed throughout the service process.",
    llm=llm,
    verbose=True
)

customer_ticket_review_task = Task(
    description="""
        Customer sent the following message request :{customer_message}
        Give the relevent reply
    """,
    expected_output="A Professional customer support reponse",
    agent=customer_support_agent
)

crew = Crew(
    agents=[customer_support_agent],
    tasks=[customer_ticket_review_task],
    verbose=True
)

if __name__ =='__main__':
    customer_message = input("Customer :")

    result = crew.kickoff(
        inputs={
            "customer_message":customer_message
        }
    )
    print(result)

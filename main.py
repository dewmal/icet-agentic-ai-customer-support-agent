from crew_ai_network import crew
if __name__ =='__main__':
    customer_message = input("Customer :")

    result = crew.kickoff(
        inputs={
            "customer_message":customer_message
        }
    )
    print(result)

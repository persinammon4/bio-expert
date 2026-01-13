from biology_expert_agent import BiologyExpertAgent

def main():
    print("Welcome to the Biology Expert Agent with PubMed!")
    expert = BiologyExpertAgent()

    while True:
        question = input("\nAsk a biology question (or 'quit' to exit): ")
        if question.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        try:
            answer = expert.answer(question)
        except Exception as e:
            answer = f"An error occurred while getting the answer: {e}"

        print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()

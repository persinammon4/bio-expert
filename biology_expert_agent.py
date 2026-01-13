from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

# Your PubMed tool wrapper
from pubmed_tool import PubMedTool

class BiologyExpertAgent:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        # Create the LLM model instance
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        # Define the PubMed tool as a LangChain tool
        @tool
        def pubmed_search(query: str) -> str:
            """Search PubMed for relevant biology papers."""
            return PubMedTool.run(query)

        # Put all tools in a list
        tools = [pubmed_search]

        # Create the agent
        # system_prompt defines the agent role/intent
        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt="You are a biology expert assistant that can call tools like PubMed search."
        )

    def answer(self, question: str) -> str:
        # Agents expect an "invoke" with messages;
        # supply a list with a single user message
        result = self.agent.invoke({"messages":[{"role":"user","content":question}]})
        return result["output"]

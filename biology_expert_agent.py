from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from pubmed_tool import PubMedTool
from ml_model import predict_protein

class BiologyExpertAgent:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        # PubMed tool
        @tool
        def pubmed_search(query: str) -> str:
            """Search PubMed for relevant biology papers based on a query."""
            return PubMedTool.run(query)

        # ML prediction tool
        @tool
        def enzyme_predictor(length: int, hydrophobicity: float, charge: int) -> str:
            """Predict whether a protein is enzymatic given its length, hydrophobicity, and charge."""
            features = {"length": length, "hydrophobicity": hydrophobicity, "charge": charge}
            return predict_protein(features)

        tools = [pubmed_search, enzyme_predictor]

        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt="You are a biology expert. You can answer questions and predict if a protein is enzymatic using enzyme_predictor."
        )

    def answer(self, question: str) -> str:
        result = self.agent.invoke({"messages":[{"role":"user","content":question}]})
        return result["output"]

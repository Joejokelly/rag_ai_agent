from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


# local model (NO OpenAI)


pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=100
)

llm = HuggingFacePipeline(pipeline=pipe)


# simple tool
def test_tool(query):
    return "Tool is working correctly"


tools = [
    Tool(
        name="TestTool",
        func=test_tool,
        description="Use this tool to test if tools are working"
    )
]


# create agent

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3

)

def main():
    print(test_tool("please test the tool"))

    ##result = agent.invoke({"input": "please test the tool"})
    ##print(result["output"])



if __name__ == "__main__":
    main()




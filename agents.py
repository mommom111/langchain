import os
os.environ["OPENAI_API_KEY"] = "" # push前確認！！
os.environ["GOOGLE_CSE_ID"] = ""
os.environ["GOOGLE_API_KEY"] = "" # push前確認！！

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, load_tools
from langchain import OpenAI, SerpAPIWrapper, LLMChain

# ツールの準備
tools = load_tools(["google-search"], llm=OpenAI())

# プロンプトテンプレートの準備
prefix = """次の質問にできる限り答えてください。次のツールにアクセスできます:"""
suffix = """始めましょう! 最終的な答えを出すときは、一人称は"ぼく"、語尾には"なのだ"を使用してください

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad"]
)

# エージェントの準備
llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

agent_executor.run("一ドル何円？")
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
   

)


model= ChatHuggingFace(llm=llm)


prompt= PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)


parser= StrOutputParser()

chain= prompt | model | parser


result= chain.invoke({'topic':'Cricket'})

print(result)

chain.get_graph().print_ascii()




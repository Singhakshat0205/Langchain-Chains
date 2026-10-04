
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, Literal

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
)

parser= StrOutputParser()

class feedback(BaseModel):
    sentiment:Literal['positive', 'negative']= Field(description='Give the sentiment of the feedback')

model= ChatHuggingFace(llm=llm)

parser2= PydanticOutputParser(pydantic_object=feedback)


prompt1= PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive, negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={
        'format_instruction':parser2.get_format_instructions()
    }
)


prompt2= PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)


prompt3= PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

classifier_chain= prompt1 | model | parser2


branch_chain= RunnableBranch(
    
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser ),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x : 'could not find sentiment')
)


chain = classifier_chain | branch_chain

result= chain.invoke({'feedback':'This is a terrible phone very negative feedback for this phone this is very bad phone '})

print(result)
chain.get_graph().print_ascii()




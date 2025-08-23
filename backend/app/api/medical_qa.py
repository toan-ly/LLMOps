from fastapi import APIRouter, HTTPException
from app.models.medical_qa import MedicalQARequest, MedicalQAResponse
from app.core.template import MEDICAL_TEMPLATE
from app.api.llm import medqa_chat_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

router = APIRouter()


class MedicalQAOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        return text

@traceable(metadata={"llm": "llama3.2-1b-4bit"})
@router.post("/medical_qa", response_model=MedicalQAResponse)
def answer_medical_question(request: MedicalQARequest):
    # Validate the number of options (must be 4)
    if len(request.choices) != 4:
        raise HTTPException(status_code=400, detail="Invalid number of choices (must be 4)")

    prompt = PromptTemplate(
        template=MEDICAL_TEMPLATE,
        input_variables=["question", "opa", "opb", "opc", "opd"]
    )

    chain = prompt | medqa_chat_llm | MedicalQAOutputParser()

    result = chain.invoke({
        "question": request.question,
        "opa": request.choices[0],
        "opb": request.choices[1],
        "opc": request.choices[2],
        "opd": request.choices[3]
    })

    return MedicalQAResponse(content=result)
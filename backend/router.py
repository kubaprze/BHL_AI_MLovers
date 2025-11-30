from fastapi import APIRouter, Depends
from db import init_db,  Issue, Device, get_session, Session, select
from pydantic import BaseModel
from typing import Annotated
from langchain_community.llms import Ollama

SessionDep = Annotated[Session, Depends(get_session)]

router = APIRouter(
    prefix="/issues"
)

try:
    # Initialize Ollama LLM
    llm = Ollama(
        model="gemma3:1b",
        base_url="http://localhost:11434",
        temperature=0.3,
        num_predict=1000
    )
    
    print("Ollama initialized")
    OLLAMA_AVAILABLE = True
    
except Exception as e:
    print(f"Ollama not available: {e}")
    print("Make sure: ollama pull model && ollama serve")
    OLLAMA_AVAILABLE = False
    llm = None

class Report(BaseModel):
    device_id: str
    description: str

@router.post("/add_device/{name}")
def add_device(name:str, session: SessionDep):

    new_device = Device(
        device_name=name
    )
    #print()
    for d in session.exec(select(Device)):
        print(d)
    session.add(new_device)

    session.commit()


@router.post("/add_issue")
def add_issue(issue: Report, session: SessionDep):

    prompt = f"""
    You Are HelpDesk Specialist.
    You analize problems with hardware and assign them severity from 1 (least severe) to 6 (most severe). 
    REPLY ONLY WITH the severity scale. 
    For example:
    - "My screen is shattered" - 6
    - "My keyboard is a little scratched" - 1

    NOW CLASSIFY THIS REQUEST:
    {issue.description}
    """
    
    result = llm.invoke(prompt)
    try:
        if int(result) in (1,2,3,4,5,6):
            result = int(result)
        else:
            result = None
    except:
        result = None

    new_issue = Issue(
        device_id=issue.device_id,
        description=issue.description,
        severity = result
    )

    session.add(new_issue)

    session.commit()


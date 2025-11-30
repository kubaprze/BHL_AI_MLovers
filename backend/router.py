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
    "The laptop casing has a small cosmetic crack" – 1
    "The mouse sometimes double-clicks unintentionally" – 2
    "The Wi-Fi disconnects a few times per day" – 3
    "The battery drains from 100% to 20% in under an hour" – 4
    "The computer freezes randomly during work" – 5
    "The device won't turn on at all" – 6

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


@router.get("/get_devices")
def get_devices(session: SessionDep):
    results = session.exec(select(Device))

    return results.all()


@router.get("/get_worst_issues")
def get_worst_issues(session: SessionDep):
    results = session.exec(
        select(Issue.description, Issue.severity, Device.device_name)
        .select_from(Issue)
        .join(Device, Device.id == Issue.device_id)
        .order_by(Issue.severity.desc())
        .limit(5)).all()

    return [
    {
        "description": r[0],
        "severity": r[1],
        "device_name": r[2]
    }
    for r in results
    ]

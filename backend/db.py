from sqlmodel import SQLModel, Field, create_engine, Session, select

class Device(SQLModel, table=True):
    id: int| None = Field(default=None, primary_key=True)
    device_name: str



class Issue(SQLModel, table=True):
    id: int | None= Field(default=None, primary_key=True)
    device_id: int = Field(default=None, foreign_key="device.id")
    description: str
    severity: int | None


engine = create_engine("sqlite:///database.db")

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session


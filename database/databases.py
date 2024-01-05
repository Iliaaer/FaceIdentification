from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base

# __sqlite_database = "sqlite:///database/myDatabase.db"
__sqlite_database = "sqlite:///database/test1.db"

__engine = create_engine(__sqlite_database, echo=False)

Base.metadata.create_all(__engine, checkfirst=True)

__DBSession = sessionmaker(bind=__engine, expire_on_commit=False)


def sql_database_reold(path_db: str):
    __sqlite_database = f"sqlite:///{path_db}"

    __engine = create_engine(__sqlite_database, echo=False)

    Base.metadata.create_all(__engine, checkfirst=True)

    __DBSession = sessionmaker(bind=__engine, expire_on_commit=False)


@contextmanager
def get_session():
    session = __DBSession()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

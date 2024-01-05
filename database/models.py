from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Users(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    last_name = Column(String)
    personal_name = Column(String)
    full_name = Column(String)
    photos = relationship("Photos", back_populates="users")

    def __init__(self, name, last_name, personal_name):
        super().__init__()
        self.name = name
        self.last_name = last_name
        self.personal_name = personal_name
        self.full_name = f"{last_name} {name} {personal_name}"


class Photos(Base):
    __tablename__ = 'photos'
    id = Column(Integer, primary_key=True)
    path = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))
    users = relationship("Users", back_populates="photos")

    def __init__(self, path, user_id):
        super().__init__()
        self.path = path
        self.user_id = user_id

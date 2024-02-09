from typing import List
from database.databases import get_session
from database.models import Users, Photos


def post_user(name: str, last_name: str, personal_name: str):
    user = Users(name=name, last_name=last_name, personal_name=personal_name)
    with get_session() as db:
        db.add(user)
        db.commit()


def get_all_users() -> List[Users]:
    with get_session() as db:
        users: Users = db.query(Users).all()
    return users


def get_user(user_id: int) -> Users:
    with get_session() as db:
        user: Users = db.query(Users).filter(Users.id == user_id).first()
    # if user:
    return user
    # return None


def post_photo(path: str, user_id: int):
    photo = Photos(path=path, user_id=user_id)
    with get_session() as db:
        db.add(photo)
        db.commit()


def get_userID_to_photo(path: str) -> int:
    with get_session() as db:
        user_id: Photos = db.query(Photos).filter(Photos.path == path).first()
    if user_id:
        return user_id.user_id
    return user_id


# post_photo("Face/01", 1)
# post_photo("Face/02", 2)
# post_photo("Face/03", 3)
# post_photo("Face/04", 4)

# ilya = get_user(get_userID_to_photo("Face/01/Ilya.png"))
# print(ilya.full_name)

# print(get_userID_to_photo("Face/01/Ilya6.png"))

# post_photo("Face/01/Ilya.png", 1)
# post_photo("Face/01/Ilya2.png", 1)
# post_photo("Face/01/Ilya3.png", 1)
# post_photo("Face/01/Ilya4.png", 1)

# ilya = get_ueser(1)
# print(ilya.full_name)

# print(get_all_users())
# for i in get_all_users():
#     print(i.id, i.full_name)

# post_user("Илья", "Зарубин", "Александрович")
# post_user("Артем", "Галеев", "Булатович")
# post_user("Анастасия", "Вирковская", "Алексеевна")
# post_user("Ольга", "Рябова", "Алексеевна")
# post_user("Евгений", "Березин", "Сергеевич")

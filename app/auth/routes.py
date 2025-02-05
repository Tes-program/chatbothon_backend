from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.user import User
from ..auth.auth_handler import AuthHandler
from pydantic import BaseModel

router = APIRouter()
auth_handler = AuthHandler()


class UserCreate(BaseModel):
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


@router.post("/signup")
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    hashed_password = auth_handler.get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Generate token
    token = auth_handler.create_access_token(str(db_user.id))
    return {"access_token": token, "token_type": "bearer"}


@router.post("/login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    # Find user
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user:
        raise HTTPException(
            status_code=401, detail="Invalid email or password")

    # Verify password
    if not auth_handler.verify_password(
        user.password, db_user.hashed_password
    ):
        raise HTTPException(
            status_code=401, detail="Invalid email or password")

    # Generate token
    token = auth_handler.create_access_token(str(db_user.id))
    return {"access_token": token, "token_type": "bearer"}

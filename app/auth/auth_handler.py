from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from requests import Session
from app.database import get_db
from app.models.user import User
from passlib.context import CryptContext


class AuthHandler:
    security = HTTPBearer()
    secret = "your-secret-key"  # Move to env variables

    def create_access_token(self, user_id: str):
        payload = {
            'exp': datetime.utcnow() + timedelta(days=7),
            'iat': datetime.utcnow(),
            'sub': user_id
        }
        return jwt.encode(payload, self.secret, algorithm='HS256')

    def verify_token(self, credentials: HTTPAuthorizationCredentials):
        try:
            payload = jwt.decode(credentials.credentials,
                                 self.secret, algorithms=['HS256'])
            return payload['sub']
        except jwt.ExpiredSignatureError:
            raise HTTPException(401, 'Token expired')
        except jwt.InvalidTokenError:
            raise HTTPException(401, 'Invalid token')

    def get_password_hash(self, password: str) -> str:
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.verify(plain_password, hashed_password)

    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db)
    ):
        user_id = self.verify_token(credentials)
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return user

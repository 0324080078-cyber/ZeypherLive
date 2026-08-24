"""ZeypherLive — SaaS Backend (Auth + Credits + API Keys)"""
import os
import json
import time
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import jwt, JWTError

app = FastAPI(title="ZeypherLive API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SECRET_KEY = os.environ.get("ZEYPHER_SECRET", "zeypher-live-secret-key-change-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 72
FREE_CREDITS = 1000
CREDIT_COST_PER_SECOND = 1

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
KEYS_FILE = os.path.join(DATA_DIR, "api_keys.json")

os.makedirs(DATA_DIR, exist_ok=True)


def _load(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


class RegisterReq(BaseModel):
    username: str
    email: str
    password: str


class LoginReq(BaseModel):
    username: str
    password: str


class CreditReq(BaseModel):
    amount: int


def _create_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def _verify_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


def _get_user(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id = _verify_token(authorization[7:])
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id


@app.post("/api/auth/register")
def register(req: RegisterReq):
    users = _load(USERS_FILE)
    if req.username in users:
        raise HTTPException(400, "Username taken")
    for u in users.values():
        if u.get("email") == req.email:
            raise HTTPException(400, "Email already registered")

    user_id = str(uuid.uuid4())[:8]
    users[req.username] = {
        "id": user_id,
        "email": req.email,
        "password": pwd_ctx.hash(req.password),
        "credits": FREE_CREDITS,
        "created": datetime.utcnow().isoformat(),
        "total_used": 0,
    }
    _save(USERS_FILE, users)

    token = _create_token(user_id)
    return {"token": token, "user_id": user_id, "credits": FREE_CREDITS, "username": req.username}


@app.post("/api/auth/login")
def login(req: LoginReq):
    users = _load(USERS_FILE)
    user = users.get(req.username)
    if not user or not pwd_ctx.verify(req.password, user["password"]):
        raise HTTPException(401, "Invalid credentials")

    token = _create_token(user["id"])
    return {"token": token, "user_id": user["id"], "credits": user["credits"], "username": req.username}


@app.get("/api/user/profile")
def profile(user_id: str = Depends(_get_user)):
    users = _load(USERS_FILE)
    for u in users.values():
        if u["id"] == user_id:
            return {
                "id": u["id"],
                "credits": u["credits"],
                "total_used": u["total_used"],
                "created": u["created"],
            }
    raise HTTPException(404, "User not found")


@app.post("/api/credits/add")
def add_credits(req: CreditReq, user_id: str = Depends(_get_user)):
    users = _load(USERS_FILE)
    for u in users.values():
        if u["id"] == user_id:
            u["credits"] += req.amount
            _save(USERS_FILE, users)
            return {"credits": u["credits"], "added": req.amount}
    raise HTTPException(404, "User not found")


@app.post("/api/credits/deduct")
def deduct_credits(amount: int = 1, user_id: str = Depends(_get_user)):
    users = _load(USERS_FILE)
    for u in users.values():
        if u["id"] == user_id:
            if u["credits"] < amount:
                raise HTTPException(402, f"Insufficient credits. Have {u['credits']}, need {amount}")
            u["credits"] -= amount
            u["total_used"] += amount
            _save(USERS_FILE, users)
            return {"credits": u["credits"], "deducted": amount}
    raise HTTPException(404, "User not found")


@app.get("/api/keys/generate")
def generate_key(user_id: str = Depends(_get_user)):
    keys = _load(KEYS_FILE)
    key = f"zl_{uuid.uuid4().hex}{uuid.uuid4().hex[:16]}"
    keys[key] = {
        "user_id": user_id,
        "created": datetime.utcnow().isoformat(),
        "active": True,
    }
    _save(KEYS_FILE, keys)
    return {"api_key": key}


@app.get("/api/keys/list")
def list_keys(user_id: str = Depends(_get_user)):
    keys = _load(KEYS_FILE)
    my_keys = [{"key": k[:12] + "...", "active": v["active"], "created": v["created"]}
               for k, v in keys.items() if v["user_id"] == user_id]
    return {"keys": my_keys}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/pricing")
def pricing():
    return {
        "free_credits": FREE_CREDITS,
        "cost_per_second": CREDIT_COST_PER_SECOND,
        "plans": [
            {"name": "Free", "credits": 1000, "price": 0},
            {"name": "Starter", "credits": 5000, "price": 5},
            {"name": "Pro", "credits": 25000, "price": 20},
            {"name": "Enterprise", "credits": 100000, "price": 75},
        ],
    }

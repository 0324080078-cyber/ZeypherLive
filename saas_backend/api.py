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
from pydantic import BaseModel
from jose import jwt, JWTError

app = FastAPI(title="ZeypherLive API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SECRET_KEY = os.environ.get("ZEYPHER_SECRET", "zeypher-live-secret-key-change-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 72
FREE_CREDITS = 1000
CREDITS_PER_SECOND = 2
ADMIN_SECRET = os.environ.get("ZEYPHER_ADMIN_SECRET", "zeypher-admin-2024")
BTC_ADDRESS = "bc1q3rq0c6j2mzz6la83t2j9mqw249fd7whyrp2u8l"

HASH_SALT = "zeypher-live-salt-2024"


def _hash_password(password: str) -> str:
    return hashlib.sha256((password + HASH_SALT).encode()).hexdigest()


def _verify_password(password: str, hashed: str) -> bool:
    return _hash_password(password) == hashed

pwd_ctx = None
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
KEYS_FILE = os.path.join(DATA_DIR, "api_keys.json")
PROOFS_FILE = os.path.join(DATA_DIR, "payment_proofs.json")

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


class ProofReq(BaseModel):
    username: str
    plan: str
    txid: str


class AdminReq(BaseModel):
    secret: str
    username: str
    credits: int


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
        "password": _hash_password(req.password),
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
    if not user or not _verify_password(req.password, user["password"]):
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


@app.post("/api/stream/start")
def stream_start(user_id: str = Depends(_get_user)):
    users = _load(USERS_FILE)
    for u in users.values():
        if u["id"] == user_id:
            if u["credits"] < 10:
                raise HTTPException(402, f"Need at least 10 credits to start. You have {u['credits']}")
            return {"credits": u["credits"], "credits_per_second": CREDITS_PER_SECOND, "max_seconds": u["credits"] // CREDITS_PER_SECOND}
    raise HTTPException(404, "User not found")


@app.post("/api/stream/tick")
def stream_tick(seconds: int = 1, user_id: str = Depends(_get_user)):
    cost = seconds * CREDITS_PER_SECOND
    users = _load(USERS_FILE)
    for u in users.values():
        if u["id"] == user_id:
            if u["credits"] < cost:
                u["credits"] = 0
                _save(USERS_FILE, users)
                raise HTTPException(402, "Out of credits")
            u["credits"] -= cost
            u["total_used"] += cost
            _save(USERS_FILE, users)
            return {"credits": u["credits"], "deducted": cost, "remaining_seconds": u["credits"] // CREDITS_PER_SECOND}
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
        "credits_per_second": CREDITS_PER_SECOND,
        "btc_address": BTC_ADDRESS,
        "plans": [
            {"id": "starter", "name": "Starter", "credits": 500, "price": 25, "description": "~4m 10s stream time"},
            {"id": "basic", "name": "Basic", "credits": 1000, "price": 45, "description": "~8m 20s stream time"},
            {"id": "plus", "name": "Plus", "credits": 2000, "price": 55, "description": "~16m 40s stream time"},
            {"id": "pro", "name": "Pro", "credits": 5000, "price": 150, "description": "~41m 40s stream time"},
            {"id": "premium", "name": "Premium", "credits": 999999, "price": 300, "description": "~8333m unlimited"},
        ],
        "notes": [
            "2 credits are deducted per second of stream time",
            "Credits never expire",
        ],
    }


PLAN_CREDITS = {
    "Starter - 500 Credits - $25": 500,
    "Basic - 1000 Credits - $45": 1000,
    "Plus - 2000 Credits - $55": 2000,
    "Pro - 5000 Credits - $150": 5000,
    "Premium - 999999 Credits - $300": 999999,
}


@app.post("/api/payment/proof")
def submit_proof(req: ProofReq):
    users = _load(USERS_FILE)
    if req.username not in users:
        raise HTTPException(404, "User not found")
    if req.plan not in PLAN_CREDITS:
        raise HTTPException(400, "Invalid plan")
    proofs = _load(PROOFS_FILE)
    proof_id = str(uuid.uuid4())[:8]
    proofs[proof_id] = {
        "user": req.username,
        "plan": req.plan,
        "txid": req.txid,
        "credits": PLAN_CREDITS[req.plan],
        "status": "pending",
        "created": datetime.utcnow().isoformat(),
    }
    _save(PROOFS_FILE, proofs)
    return {"message": "Payment proof submitted. Admin will verify and add credits within 24 hours.", "proof_id": proof_id}


@app.get("/api/admin/proofs")
def list_proofs(secret: str = Header(None)):
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Unauthorized")
    proofs = _load(PROOFS_FILE)
    pending = {k: v for k, v in proofs.items() if v["status"] == "pending"}
    return {"proofs": pending}


@app.post("/api/admin/approve")
def approve_proof(req: AdminReq):
    if req.secret != ADMIN_SECRET:
        raise HTTPException(403, "Unauthorized")
    proofs = _load(PROOFS_FILE)
    users = _load(USERS_FILE)
    for pid, proof in proofs.items():
        if proof["user"] == req.username and proof["status"] == "pending":
            proof["status"] = "approved"
            proof["approved_at"] = datetime.utcnow().isoformat()
            _save(PROOFS_FILE, proofs)
            if req.username in users:
                users[req.username]["credits"] += proof["credits"]
                _save(USERS_FILE, users)
                return {"message": f"Added {proof['credits']} credits to {req.username}", "credits": users[req.username]["credits"]}
    raise HTTPException(404, "No pending proof found for this user")


@app.post("/api/admin/deny")
def deny_proof(req: AdminReq):
    if req.secret != ADMIN_SECRET:
        raise HTTPException(403, "Unauthorized")
    proofs = _load(PROOFS_FILE)
    for pid, proof in proofs.items():
        if proof["user"] == req.username and proof["status"] == "pending":
            proof["status"] = "denied"
            proof["denied_at"] = datetime.utcnow().isoformat()
            _save(PROOFS_FILE, proofs)
            return {"message": f"Denied proof for {req.username}"}
    raise HTTPException(404, "No pending proof found for this user")


@app.post("/api/admin/add_credits")
def admin_add_credits(req: AdminReq):
    if req.secret != ADMIN_SECRET:
        raise HTTPException(403, "Unauthorized")
    users = _load(USERS_FILE)
    if req.username not in users:
        raise HTTPException(404, "User not found")
    users[req.username]["credits"] += req.credits
    _save(USERS_FILE, users)
    return {"credits": users[req.username]["credits"], "added": req.credits}

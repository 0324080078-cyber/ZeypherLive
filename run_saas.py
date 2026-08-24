"""ZeypherLive — SaaS Server Launcher"""
import os
import sys
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))

from saas_backend.api import app
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

DASHBOARD = os.path.join(os.path.dirname(__file__), "saas_backend", "dashboard.html")


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(DASHBOARD)


@app.get("/")
def root():
    return FileResponse(DASHBOARD)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[ZeypherLive SaaS] Starting on port {port}")
    print(f"[ZeypherLive SaaS] Dashboard: http://localhost:{port}/dashboard")
    print(f"[ZeypherLive SaaS] API: http://localhost:{port}/api/health")
    uvicorn.run(app, host="0.0.0.0", port=port)

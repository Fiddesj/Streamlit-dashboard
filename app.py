"""
Root launcher for AI Development Rate Streamlit Dashboard
Delegates directly to Compute subindex/dashboard.py
"""
import os
import sys

# Path to dashboard
dashboard_path = os.path.join(os.path.dirname(__file__), "Compute subindex", "dashboard.py")

with open(dashboard_path, "r", encoding="utf-8") as f:
    code = f.read()

# Execute dashboard in current context
exec(compile(code, dashboard_path, 'exec'))


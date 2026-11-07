"""
IncidentX Backend Environment Verification Script
==================================================
Verifies:
1. Active Python interpreter is the project's .venv.
2. All required dependencies are installed without broken packages.
3. Core libraries (FastAPI, LangGraph, OpenAI, FAISS, NumPy, Pydantic) import and function cleanly.
4. Python environment and platform compatibility (version, 64-bit, FAISS 1536-dim vector indexing).

Usable as:
  - Standalone script: python backend/verify_env.py
  - Pytest test suite: pytest backend/verify_env.py
"""

import os
import sys
import platform
import importlib
import importlib.metadata
from pathlib import Path
from typing import List, Tuple, Dict, Any


def get_backend_dir() -> Path:
    """Return backend directory path."""
    current = Path(__file__).resolve().parent
    return current


def get_requirements_path() -> Path:
    """Locate requirements.txt file."""
    backend_dir = get_backend_dir()
    candidate = backend_dir / "requirements.txt"
    if candidate.exists():
        return candidate
    candidate_root = backend_dir.parent / "backend" / "requirements.txt"
    if candidate_root.exists():
        return candidate_root
    raise FileNotFoundError("requirements.txt could not be found.")


def check_virtual_environment() -> Tuple[bool, str, Dict[str, Any]]:
    """Verify that Python is running from the project's .venv."""
    in_venv = sys.prefix != sys.base_prefix
    exe_path = Path(sys.executable).resolve()
    venv_dir = get_backend_dir() / ".venv"
    
    # Check if executable is inside the project's .venv directory
    is_project_venv = False
    try:
        exe_path.relative_to(venv_dir)
        is_project_venv = True
    except ValueError:
        is_project_venv = ".venv" in exe_path.parts

    details = {
        "executable": str(exe_path),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "python_version": platform.python_version(),
        "architecture": platform.architecture()[0],
        "is_venv": in_venv,
        "is_project_venv": is_project_venv
    }

    if not in_venv:
        msg = (
            f"FAIL: Running under system Python ({exe_path}), not a virtual environment.\n"
            f"Activate the environment with: .\\backend\\.venv\\Scripts\\Activate.ps1\n"
            f"Or run with: .\\backend\\.venv\\Scripts\\python.exe"
        )
        return False, msg, details

    if not is_project_venv:
        msg = f"WARN: Running in a virtual environment, but not at expected project path ({venv_dir})."
        return False, msg, details

    msg = f"PASS: Active Python interpreter is project .venv ({exe_path})"
    return True, msg, details


def check_installed_dependencies() -> Tuple[bool, str, List[str]]:
    """Verify that all packages listed in requirements.txt are installed."""
    req_file = get_requirements_path()
    with open(req_file, "r", encoding="utf-8") as f:
        lines = [line.strip().lstrip("\ufeff") for line in f if line.strip() and not line.startswith("#")]

    missing = []
    installed_count = 0
    for req in lines:
        pkg_name = req.split("==")[0].split(">=")[0].split("<=")[0].strip()
        try:
            importlib.metadata.distribution(pkg_name)
            installed_count += 1
        except importlib.metadata.PackageNotFoundError:
            missing.append(req)

    if missing:
        msg = f"FAIL: {len(missing)} package(s) missing from environment: {', '.join(missing)}"
        return False, msg, missing

    msg = f"PASS: All {installed_count} packages in requirements.txt are installed."
    return True, msg, []


def check_core_imports_and_functionality() -> Tuple[bool, str, Dict[str, str]]:
    """Verify successful imports and functional smoke tests for core dependencies."""
    results = {}
    failures = []

    # 1. NumPy
    try:
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        assert norm > 0
        results["numpy"] = f"v{np.__version__} (matrix/norm OK)"
    except Exception as e:
        results["numpy"] = f"FAILED: {e}"
        failures.append(f"numpy: {e}")

    # 2. FAISS (including OpenAI text-embedding-3-small 1536-dim vector indexing)
    try:
        import faiss
        import numpy as np
        dim = 1536
        idx = faiss.IndexFlatL2(dim)
        data = np.random.randn(5, dim).astype("float32")
        faiss.normalize_L2(data)
        idx.add(data)
        dists, nbrs = idx.search(data[:1], 2)
        assert idx.ntotal == 5
        assert nbrs.shape == (1, 2)
        results["faiss"] = f"v{getattr(faiss, '__version__', 'cpu')} (1536-dim IndexFlatL2 search OK)"
    except Exception as e:
        results["faiss"] = f"FAILED: {e}"
        failures.append(f"faiss: {e}")

    # 3. FastAPI
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI(title="IncidentX Health Check")

        @app.get("/ping")
        def ping():
            return {"status": "ok"}

        client = TestClient(app)
        res = client.get("/ping")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}
        import fastapi
        results["fastapi"] = f"v{fastapi.__version__} (TestClient app invocation OK)"
    except Exception as e:
        results["fastapi"] = f"FAILED: {e}"
        failures.append(f"fastapi: {e}")

    # 4. LangGraph
    try:
        from typing import TypedDict
        from langgraph.graph import StateGraph, START, END

        class FlowState(TypedDict):
            counter: int

        def increment_step(state: FlowState) -> dict:
            return {"counter": state["counter"] + 1}

        builder = StateGraph(FlowState)
        builder.add_node("increment", increment_step)
        builder.add_edge(START, "increment")
        builder.add_edge("increment", END)
        graph = builder.compile()

        res = graph.invoke({"counter": 41})
        assert res["counter"] == 42
        import langgraph
        results["langgraph"] = f"v{getattr(langgraph, '__version__', 'installed')} (StateGraph execution OK)"
    except Exception as e:
        results["langgraph"] = f"FAILED: {e}"
        failures.append(f"langgraph: {e}")

    # 5. OpenAI
    try:
        import openai
        client = openai.OpenAI(api_key="mock-key-for-environment-verification")
        assert client is not None
        results["openai"] = f"v{openai.__version__} (Client instantiation OK)"
    except Exception as e:
        results["openai"] = f"FAILED: {e}"
        failures.append(f"openai: {e}")

    # 6. Pydantic
    try:
        import pydantic
        from pydantic import BaseModel, Field

        class VerificationModel(BaseModel):
            check_name: str
            passed: bool
            score: float = Field(ge=0.0, le=1.0)

        item = VerificationModel(check_name="env_test", passed=True, score=0.99)
        assert item.passed is True
        results["pydantic"] = f"v{pydantic.__version__} (Schema validation OK)"
    except Exception as e:
        results["pydantic"] = f"FAILED: {e}"
        failures.append(f"pydantic: {e}")

    if failures:
        msg = f"FAIL: Functional check failures: {'; '.join(failures)}"
        return False, msg, results

    msg = "PASS: All core libraries imported and passed functional verification."
    return True, msg, results


def check_compatibility() -> Tuple[bool, str, Dict[str, Any]]:
    """Check Python environment and platform compatibility."""
    version_info = sys.version_info
    arch = platform.architecture()[0]
    os_name = platform.system()
    
    # IncidentX requires Python >= 3.10 and AMD64 architecture for FAISS binaries
    is_version_compatible = version_info >= (3, 10)
    is_64bit = arch == "64bit"

    details = {
        "python_version": f"{version_info.major}.{version_info.minor}.{version_info.micro}",
        "architecture": arch,
        "os": os_name,
        "is_version_compatible": is_version_compatible,
        "is_64bit": is_64bit
    }

    if not is_version_compatible:
        return False, f"FAIL: Python {version_info.major}.{version_info.minor} is unsupported. Python 3.10+ required.", details

    if not is_64bit:
        return False, "FAIL: 64-bit Python architecture required for FAISS and numerical libraries.", details

    msg = f"PASS: Python {version_info.major}.{version_info.minor}.{version_info.micro} ({arch}, {os_name}) is fully compatible."
    return True, msg, details


# --- Pytest test definitions ---

def test_active_interpreter_is_venv():
    ok, msg, _ = check_virtual_environment()
    assert ok, msg


def test_dependencies_installed():
    ok, msg, _ = check_installed_dependencies()
    assert ok, msg


def test_core_imports_and_functionality():
    ok, msg, _ = check_core_imports_and_functionality()
    assert ok, msg


def test_environment_compatibility():
    ok, msg, _ = check_compatibility()
    assert ok, msg


# --- CLI Runner ---

def run_all_checks() -> int:
    header = "=" * 60
    print(header)
    print(" IncidentX Backend Environment Verification")
    print(header)

    all_passed = True

    # 1. Interpreter check
    print("\n[1/4] Checking Active Python Interpreter...")
    ok1, msg1, det1 = check_virtual_environment()
    print(f"      Executable: {det1['executable']}")
    print(f"      Version:    {det1['python_version']} ({det1['architecture']})")
    print(f"      Result:     {msg1}")
    if not ok1:
        all_passed = False

    # 2. Dependency installation check
    print("\n[2/4] Checking Requirements & Dependencies...")
    ok2, msg2, missing = check_installed_dependencies()
    print(f"      Result:     {msg2}")
    if not ok2:
        all_passed = False

    # 3. Core module import and smoke check
    print("\n[3/4] Checking Core Packages (FastAPI, LangGraph, OpenAI, FAISS, NumPy, Pydantic)...")
    ok3, msg3, components = check_core_imports_and_functionality()
    for name, status in components.items():
        print(f"      - {name:<10}: {status}")
    print(f"      Result:     {msg3}")
    if not ok3:
        all_passed = False

    # 4. Compatibility check
    print("\n[4/4] Checking Python Environment Compatibility...")
    ok4, msg4, det4 = check_compatibility()
    print(f"      OS / Arch:  {det4['os']} / {det4['architecture']}")
    print(f"      Result:     {msg4}")
    if not ok4:
        all_passed = False

    print("\n" + header)
    if all_passed:
        print(" [SUCCESS] All backend environment checks passed.")
        print(header)
        return 0
    else:
        print(" [FAILURE] One or more environment checks failed. Review output above.")
        print(header)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_checks())

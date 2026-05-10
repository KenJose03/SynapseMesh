import docker

def python_sandbox_tool(code: str):
    """Section 2: Production-grade isolated code execution."""
    client = docker.from_env()
    try:
        # Spin up a tiny alpine container, run code, and kill it immediately
        container = client.containers.run(
            "python:3.10-alpine",
            command=f"python3 -c '{code}'",
            remove=True,
            stdout=True,
            stderr=True,
            mem_limit="128m", # Constraint
            timeout=5
        )
        return {"status": "success", "stdout": container.decode()}
    except Exception as e:
        return {"status": "error", "stderr": str(e)}
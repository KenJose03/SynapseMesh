import docker

def python_sandbox_tool(code: str):
    """Section 2: Production-grade isolated code execution."""
    try:
        client = docker.from_env()
        # Spin up a tiny isolated alpine container, run code with no network, and remove immediately
        output = client.containers.run(
            "python:3.10-alpine",
            command=["python3", "-c", code],
            remove=True,
            stdout=True,
            stderr=True,
            mem_limit="128m",
            network_disabled=True,
        )
        return {"status": "success", "stdout": output.decode("utf-8", errors="replace")}
    except Exception as e:
        return {"status": "error", "stderr": str(e)}
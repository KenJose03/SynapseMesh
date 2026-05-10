from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ToolLog(BaseModel):
    """Logs for Section 2: Tool Calling & Failure Modes"""
    tool_name: str
    input_data: str
    output_data: str
    latency: float
    status: str  # e.g., 'success', 'timeout', 'malformed_input'
    accepted_by_agent: bool = True 

class AgentAction(BaseModel):
    """Logs for Section 1: Orchestration Reasoning"""
    agent_id: str
    thought_process: str  # The structured reasoning for the decision
    output: str
    token_count: int
    timestamp: datetime = Field(default_factory=datetime.now)

class SharedContext(BaseModel):
    """The 'Shared Context Object' required by Section 1"""
    job_id: str
    original_query: str
    
    # Section 1: Sub-tasks and Dependency Graphs
    task_graph: List[Dict[str, Any]] = [] 
    
    # Section 3: Context Budgeting
    total_tokens_used: int = 0
    remaining_budget: int = 4000
    
    # Memory and Provenance
    history: List[AgentAction] = []
    tool_vault: List[ToolLog] = []  # Lossless storage
    final_answer: Optional[str] = None
    
    def add_tokens(self, count: int):
        self.total_tokens_used += count
        self.remaining_budget -= count
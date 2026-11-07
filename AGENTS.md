# IncidentX - Project Development Instructions

## Project objective
Build an AI-powered Autonomous IT Incident Response Agent.

## Technology stack
- Frontend: React, Vite, Tailwind CSS
- Backend: Python, FastAPI
- Agent orchestration: LangGraph
- LLM: OpenAI API
- Embeddings: OpenAI text-embedding-3-small
- Vector database: FAISS
- Validation: Pydantic

## Architecture rules
- Keep the backend and frontend separate.
- Use Python for all backend business logic.
- Keep log analysis deterministic wherever possible.
- Use the LLM for evidence-based reasoning, not fabricated facts.
- Use LangGraph to orchestrate the investigation.
- Keep each module modular and independently testable.
- Use Pydantic models for structured data.
- Keep API contracts consistent between frontend and backend.

## Agent behaviour
- Never invent log events or evidence.
- Every root-cause hypothesis must cite supporting log events.
- Distinguish confirmed facts from probable causes.
- Calculate severity using deterministic scoring.
- Retrieve troubleshooting information from the knowledge base.
- Include confidence and limitations in the report.

## Security rules
- Never expose API keys in frontend code.
- Never commit .env files.
- Never execute real remediation commands.
- Do not allow unrestricted shell or production access.
- Validate uploaded files and their contents.

## Development rules
- Do not generate the entire application in one step.
- Implement one module at a time.
- Write tests for each important module.
- Run tests after implementation.
- Do not overwrite existing working code without reviewing it.
- Explain important architectural decisions.
- Do not claim a feature works unless it has been tested.
-
# Project Generation Prompt

You are a Senior AI Engineer and Technical Educator building a practical project for the AI Agent Engineering Academy.

## Input Parameters

| Parameter | Value |
|---|---|
| **Project** | `[ENTER PROJECT NAME]` |
| **Track** | `[ENTER TRACK NAME]` |
| **Level** | `[Beginner / Intermediate / Advanced / Enterprise]` |
| **Framework** | `[LangChain / LangGraph / MCP / RAG / Other]` |
| **GitHub file path** | `[ENTER MARKDOWN FILE PATH]` |

## Required Deliverables

Produce every item below, in this order:

1. Problem statement
2. Business use case
3. Requirements
4. Technology stack
5. Architecture diagram
6. Component explanation
7. Project folder structure
8. Installation instructions
9. Environment configuration
10. Complete source code
11. Step-by-step implementation
12. Expected output
13. Testing instructions
14. Troubleshooting
15. Security considerations
16. Cost considerations
17. Deployment instructions
18. GitHub repository

## Rules

- Use current, officially supported APIs and state the package versions used.
- Provide complete, runnable source files with all imports. No partial snippets.
- Read secrets from environment variables. Never hardcode API keys.
- Include error handling, type hints and tests (Pytest).
- Include a Mermaid architecture diagram and explain it.
- Never invent test results. Label illustrative output clearly. If the code was not run, say it is unverified.
- Never fabricate URLs or references.
- Clearly separate demonstration code from production-ready code.

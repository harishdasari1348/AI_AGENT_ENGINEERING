# Content Guidelines

These rules apply to every article and project in the academy.

## Part 1: Standard Blog Writing Template

Every article should follow this format.

### Article Metadata

```yaml
Title:
Track:
Module:
Difficulty:
Estimated reading time:
Prerequisites:
Technologies:
Framework versions:
Last verified date:
```

### Article Structure

| # | Section | Purpose |
|---|---|---|
| 1 | **Introduction** | Explain the topic and why it matters. |
| 2 | **Learning Objectives** | List what the reader will learn. |
| 3 | **Prerequisites** | Explain required knowledge. |
| 4 | **Conceptual Explanation** | Start with simple definitions and practical analogies. |
| 5 | **Technical Deep Dive** | Explain internal architecture and important components. |
| 6 | **Architecture Diagram** | Use Mermaid or another clear diagram format. |
| 7 | **Installation and Setup** | Include exact installation commands. |
| 8 | **Project Structure** | Explain each directory and file. |
| 9 | **Implementation** | Provide complete working source code. |
| 10 | **Code Explanation** | Explain the important statements, functions, parameters and execution flow. |
| 11 | **Execution** | Explain how to run the project. |
| 12 | **Expected Output** | Show realistic terminal output or application results. |
| 13 | **Internal Workflow** | Explain what happens behind the scenes. |
| 14 | **Troubleshooting** | Cover common installation, configuration and runtime errors. |
| 15 | **Real-World Applications** | Explain practical industry use cases. |
| 16 | **Best Practices** | Explain maintainability, security and performance. |
| 17 | **Hands-on Exercises** | Provide tasks for learners. |
| 18 | **Interview Questions** | Include beginner, intermediate and advanced questions. |
| 19 | **Summary** | Recap the main concepts. |
| 20 | **Next Article** | Provide the next lesson and its prerequisites. |
| 21 | **References** | Link to official documentation and reliable resources. |

## Part 2: Content Quality Standards

To maintain educational quality, every article must satisfy the following standards.

### Technical Accuracy

- Use official documentation.
- Verify framework versions.
- Avoid outdated APIs unless teaching historical approaches.
- Run code examples before publishing.
- Include complete imports.
- Verify package compatibility.
- Never invent expected outputs.

### Beginner Friendliness

- Explain every new technical term.
- Avoid assuming previous AI knowledge.
- Introduce one concept at a time.
- Provide simple examples before complex examples.
- Explain why a particular approach is used.

### Code Quality

- Use clear naming conventions.
- Include type hints where useful.
- Handle exceptions.
- Avoid hardcoded credentials.
- Use environment variables.
- Include dependency files.
- Follow Python best practices.

### Security

- Never publish real API keys.
- Avoid exposing private data.
- Discuss prompt injection.
- Explain tool permissions.
- Include human approval where appropriate.
- Distinguish demonstrations from production-ready implementations.

### Originality

- Write original explanations.
- Create original examples.
- Design original diagrams.
- Attribute external references.
- Do not reproduce third-party course materials.

## Part 3: Curriculum Maintenance Rules

AI frameworks change frequently. The educational platform must remain current.

For each published tutorial:

- Record the tested Python version.
- Record framework and package versions.
- Include the date last tested.
- Verify APIs against official documentation.
- Test all installation commands.
- Revisit tutorials after major framework updates.
- Clearly label deprecated approaches.
- Keep project dependencies reproducible.
- Update broken examples.
- Maintain a changelog for important revisions.

> ⚠️ **Do not claim a project is production-ready merely because it runs locally.**

## Part 4: File and Naming Conventions

- Track folders are numbered and uppercase, for example `03-LANGCHAIN/`.
- Module folders and topic files are numbered, lowercase and hyphenated, for example `03-models-and-messages/01-what-is-a-chat-model.md`.
- Every topic file starts with YAML front matter (`title`, `description`, `track`, `module`, `difficulty`, `status`, `tags`).
- Change `status` from `placeholder` to `published` when the article is complete.
- Use relative links only, and only to files that exist.

# Contributing

Thank you for helping build the AI Agent Engineering Academy.

## Writing a Lesson

1. Pick a placeholder topic from [the roadmap](docs/roadmap.md).
2. Generate a draft with [the blog generation master prompt](docs/prompts/BLOG_GENERATION_MASTER_PROMPT.md), using the topic's file path.
3. Review the draft against [the content guidelines](docs/content-guidelines.md): verify APIs, versions and every reference against official documentation.
4. Run every code example. If you could not run it, label it as unverified.
5. Replace all `TODO` markers and set `status: "published"` in the front matter.
6. Tick the topic in [the roadmap](docs/roadmap.md).
7. Run `mkdocs build --strict` and fix any warnings before opening a pull request.

## Building a Project

1. Pick a project from [the project list](docs/projects.md).
2. Use [the project generation prompt](docs/prompts/PROJECT_GENERATION_PROMPT.md).
3. Cover all 18 required deliverables.
4. Review the code with [the code review prompt](docs/prompts/CODE_REVIEW_PROMPT.md).

## Adding a Page to the Website

New Markdown files must also be added to the `nav` section of `mkdocs.yml`, or they will not appear in the site navigation.

## Rules

- Never commit real API keys or secrets. Use `.env` files (already ignored by `.gitignore`).
- Do not rename or renumber existing files; other pages link to them.
- Do not add topics that are not in the roadmap without updating [docs/original-roadmap.md](docs/original-roadmap.md) first.
- Write original explanations, examples and diagrams. Do not copy third-party course material.

## Commit Messages

Use short, descriptive messages, for example `docs(langchain): write models-and-messages lessons`.

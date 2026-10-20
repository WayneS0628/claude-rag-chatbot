# Claude RAG Chatbot

Hey Quora team I made this repo just to host this little project for you to see.

If you're a recruiter from another company... well, this is kinda awkward. Hi! Please stay. It's still a good project.

## What it is

A simple RAG (Retrieval-Augmented Generation) chatbot built in Streamlit on the Claude API. Ask it about Harvard, Cornell, or Duke and it will:

1. **Retrieve**: a fast Haiku call acts as a router and picks which knowledge base file is relevant
2. **Augment**: that file gets dropped into the system prompt
3. **Generate**: Claude streams back an answer grounded in the document

Ask it about anything else and it just behaves like a normal chatbot. It does not pretend to know things about your school. It has boundaries.

Built as part of my Break Through Tech AI fellowship. The original course exercise was written for OpenAI, so I ported it to Anthropic because I wanted to actually learn the Claude API.

## The part that fought back

Streamlit reruns the entire script every time you do anything, so the chat history has to live in session state. Meanwhile, Anthropic's API insists the conversation starts with a user message. My friendly greeting message was breaking the very first question. Classic. The fix: strip the greeting before each API call and roll back any unanswered turns if a call fails.

## Run it yourself

```bash
pip install streamlit anthropic python-dotenv
```

Create a `.env` file next to the script:

```
ANTHROPIC_API_KEY=your-key-here
```

Folder layout:

```
unit07_08_RAG.py
.env
data/knowledge_base/harvard.txt
data/knowledge_base/cornell.txt
data/knowledge_base/duke.txt
```

Then:

```bash
streamlit run unit07_08_RAG.py
```

## What I'd do next

Right now it sends the whole document every turn. Once the files get big, the router gets replaced with chunking, embeddings, and top-k similarity search. Steps 2 and 3 stay exactly the same.

---

Made by Wayne Simmons · [waynesimmonsjr.com](https://waynesimmonsjr.com)

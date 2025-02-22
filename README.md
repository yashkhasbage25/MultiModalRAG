# MultiModal RAG in LangChain

Repo for code of a DS@M medium blog. 
Blog: [Building a multimodal Retrieval Augmented Generation (RAG) system: A technical overview | by Yash Khasbage | Data Science at Microsoft | Feb, 2025 | Medium](https://medium.com/p/032d0ecd81a9) 

# Running the code

1. Keep your testing PDFs in a folder `papers` in the root.
2. Create a deployment of GPT-4o from the azure portal.
3. Fill up the `azure_oai.env` environment file with the credentials of the 4o model deployed in Azure.
4. Create a python venv
```sh
python -m venv .venv
# activate the venv
# Linux/MacOS
source .venv/bin/activate
# Windows
.venv/Scripts/activate
```
5. Install the dependencies
```sh
pip install -r requirements.txt
```
6. Run the RAG and observe the output on terminal
```sh
python mm_rag.py
```


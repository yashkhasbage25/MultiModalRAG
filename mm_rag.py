"""
Author: Yash Khasbage
Email: yashkhasbage25@gmail.com
Description: Multi-modal RAG (Retrieval Augmented Generation) model using OpenAI GPT-4o and Azure OpenAI Embeddings.
"""

import os
import sys
import uuid
import shutil
import base64
import os.path as osp
from enum import Enum, auto
from operator import itemgetter
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import List, Union, ByteString
from unstructured.partition.pdf import partition_pdf

from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever


load_dotenv("azure_oai.env")
shutil.rmtree("./chroma_db", ignore_errors=True)

gpt4o = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    deployment_name="gpt-4o",
    model_name="gpt-4o",
    api_version=os.getenv("OPENAI_API_VERSION")
)

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_deployment="text-embed-ada-002",
    model="text-embedding-ada-002",
    api_version=os.getenv("OPENAI_API_VERSION")
)


class Summarizer:
    """
    Summarizer class to summarize text, tables and images using GPT-4o model.
    """
    @staticmethod
    def summarize_text(text: str) -> str:
        """Summarize text using GPT-4o model.

        Args:
            text (str): Text to summarize

        Returns:
            str: Summarized text
        """
        prompt = f'''
You are given a text. Summarize it in a few sentences for semantic retrieval.
Do not include any additional words like Summary: etc.
---
Here is the text:
{text}
'''
        try:
            response: BaseMessage = gpt4o.invoke([
                HumanMessage(content=[
                    {'type': 'text', 'text': prompt},
                ])
            ])
            return response.content
        except Exception as e:
            print(f"Error in Summarizer.summarize_text {e}")
            return None

        
    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode image to base64.

        Args:
            image_path (str): Path to image

        Returns:
            str: Base64 encoded image
        """        
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def summarize_image(image_path: str) -> str:
        """Summarize image using GPT-4o model.

        Args:
            image_path (str): Path to image

        Returns:
            str: Summarized image
        """        
        prompt = '''
You are given a image. Summarize the image for semantic retrieval. 
Do not include words like "Summary: etc.
'''
        assert osp.exists(image_path), f"Image path does not exist {image_path}"
        base64_image = Summarizer.encode_image(image_path)
        try:
            response: BaseMessage = gpt4o.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ])
            ])
            return response.content
        except Exception as e:
            print(f"Error in Summarizer.summarize_image {e}")
            return None
    
    @staticmethod
    def summarize_table(table: str) -> str:
        """Summarize table using GPT-4o model.

        Args:
            table (str): Table to summarize

        Returns:
            str: Summarized table
        """        
        prompt = f'''
You are given a table. Summarize the table for semantic retrieval. 
Do not include any additional words like Summary: etc.
---
Here is the table:
{table}
'''
        try:
            response: BaseMessage = gpt4o.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                ])
            ])
            return response.content
        except Exception as e:
            print(f"Error in Summarizer.summarize_table {e}")
            return None


class RAGDataType(Enum):
    """
    Data types for RAG model.
    """    
    TEXT = auto()
    TABLE = auto()
    IMAGE = auto()


@dataclass
class DataInstance:
    """
    Data class for data instance to be stored in db.
    """
    data_type: RAGDataType
    data: Union[ByteString, str]


@dataclass
class DataSummaryInstance:
    """
    Data class for summarized data instance.
    """
    data_type: RAGDataType
    data: Union[ByteString, str]
    summary: str


class DataIngestor:
    """
    Data Ingestor class to ingest data from documents.
    """
    def __init__(self, docs_dir: str, image_output_dir_path: str = 'figures'):
        """Initialize DataIngestor class.

        Args:
            docs_dir (str): Path to documents directory
            image_output_dir_path (str, optional): Directory for storing extracted images. Defaults to 'figures'.
        """
        self.docs_dir: str = docs_dir
        self.image_output_dir_path = image_output_dir_path
        self.document_paths: List[str] = []
        shutil.rmtree(self.image_output_dir_path, ignore_errors=True)

    def locate_data(self):
        """
        Locate documents in the directory.
        """
        for root, dirs, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith(".pdf"):
                    self.document_paths.append(os.path.join(root, file))
        print("Located documents: ", len(self.document_paths))

    def extract_text_tables_images(self) -> List[DataInstance]:
        """
        Extract text, tables and images from documents.
        """
        data_instances: List[DataInstance] = []
        for doc_path in self.document_paths:
            if doc_path.endswith(".pdf"):
                pdf_elements = partition_pdf(
                    filename=doc_path,
                    extract_images_in_pdf=True,
                    infer_table_structure=True,
                    chunking_strategy="by_title",
                    strategy='hi_res',
                    mode='elements',
                    max_characters=4000,
                    image_output_dir_path=self.image_output_dir_path
                )
                for element in pdf_elements:
                    if element.category == 'Table':
                        data_instances.append(DataInstance(RAGDataType.TABLE, element.text))
                    elif element.category == 'CompositeElement':
                        data_instances.append(DataInstance(RAGDataType.TEXT, element.text))
                    else:
                        print('Unsupported element category: ', element.category)

        for file in os.listdir(self.image_output_dir_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                data_instances.append(DataInstance(RAGDataType.IMAGE, osp.join(self.image_output_dir_path, file)))
        print("Extracted data instances: ", len(data_instances))
        return data_instances
    
    def summarize_text_tables_images(self, data_instances: List[DataInstance]) -> List[str]:
        """Summarize text, tables and images.

        Args:
            data_instances (List[DataInstance]): List of data instances

        Raises:
            ValueError: Unsupported data type

        Returns:
            List[str]: List of summarized data instances
        """        
        summaries: List[DataSummaryInstance] = []
        datatype_counts = {RAGDataType.TEXT: 0, RAGDataType.TABLE: 0, RAGDataType.IMAGE: 0}
        for data_instance in data_instances:
            summary: str = None
            if data_instance.data_type == RAGDataType.TEXT:
                summary = Summarizer.summarize_text(data_instance.data)
            elif data_instance.data_type == RAGDataType.TABLE:
                summary = Summarizer.summarize_table(data_instance.data)
            elif data_instance.data_type == RAGDataType.IMAGE:
                summary = Summarizer.summarize_image(data_instance.data)
            else:
                raise ValueError("Unsupported data type")
            if summary is not None:
                summaries.append(
                    DataSummaryInstance(
                        data_instance.data_type,
                        data_instance.data,
                        summary
                    )
                )
                datatype_counts[data_instance.data_type] += 1
        print("Summarized data instances: ", len(summaries))
        print("Data type counts: ", datatype_counts)
        return summaries


class Retriever:
    """
    Retriever class to ingest data and retrieve data from db
    """
    def __init__(self):
        """
        Initialize Retriever class.
        """        
        self.data_ingestor: DataIngestor = None
        self.vector_db: Chroma = None
        self.doc_db: InMemoryStore = None
        self.retriever: MultiVectorRetriever = None

    def ingest_data(
            self, docs_dir: str, 
            collection_name: str = "mm_rag"
        ) -> None:
        """Ingest data into db.

        Args:
            docs_dir (str): Path to documents directory
            collection_name (str, optional): Name of collection in db. Defaults to "mm_rag".
        """        
        self.vector_db: Chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory="./chroma_db",
            create_collection_if_not_exists=True
        )
        self.doc_db = InMemoryStore()
        self.data_ingestor = DataIngestor(docs_dir)
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_db,
            docstore=self.doc_db,
        )
        self.data_ingestor.locate_data()
        data_instances: List[DataInstance] = self.data_ingestor.extract_text_tables_images()
        if len(data_instances) == 0:
            print("No data instances found, exiting")
            sys.exit(0)
        data_summaries: List[DataSummaryInstance] = self.data_ingestor.summarize_text_tables_images(data_instances)
        
        self._ingest_data_into_db(data_summaries)

    def _ingest_data_into_db(self, data_summaries: List[DataSummaryInstance]):
        ids = [str(uuid.uuid4()) for _ in data_summaries]
        summary_docs = [
            Document(
                page_content=d.summary, 
                metadata={
                    'doc_id': ids[i], 
                    "data_type": d.data_type.value
                }
            ) for i, d in enumerate(data_summaries)
        ]
        docs = [
            (ids[i], Document(
                page_content=d.data, 
                metadata={
                    'doc_id': ids[i],
                    "data_type": d.data_type.value
                }
            )) for i, d in enumerate(data_summaries)
        ]
        print("Adding documents to db")
        print("Number of summary documents to add: ", len(summary_docs))
        print("Number of documents to add: ", len(docs))
        self.retriever.vectorstore.add_documents(summary_docs)
        self.retriever.docstore.mset(docs)
        print("Data ingested into db")

class MultiModalRAG:
    """
    Multi-modal RAG class to answer queries using context documents.
    """
    def __init__(self, docs_dir: str):
        """Initialize MultiModalRAG class.

        Args:
            docs_dir (str): Path to documents directory
        """
        self.retriver: Retriever = Retriever()
        self.retriver.ingest_data(docs_dir)
        self.retriever_chain = (
            itemgetter('query')
                |
            self.retriver.retriever
        )

    @staticmethod
    def create_query_context_prompt(args) -> List[HumanMessage]:
        """
        Create query context prompt.
        """
        print("Creating query context prompt")
        query: str = args['query']
        retrieved_data: List[Document] = args['retrieved_data']
        messages: List[str] = []
        text_docs: List[str] = [doc.page_content for doc in retrieved_data if doc.metadata['data_type'] == RAGDataType.TEXT.value]
        table_docs: List[str] = [doc.page_content for doc in retrieved_data if doc.metadata['data_type'] == RAGDataType.TABLE.value]
        imgs_docs: List[str] = [doc.page_content for doc in retrieved_data if doc.metadata['data_type'] == RAGDataType.IMAGE.value]
        print("Retrieved data: ", len(retrieved_data))
        print("Text docs: ", len(text_docs))
        print("Table docs: ", len(table_docs))
        print("Image docs: ", len(imgs_docs))
        text_message: dict = {
            'type': 'text',
            'text': f"""
You are given a query and you need to answer the query using the context documents (text, tables and images) below.
Query: {query}

Context documents:
{"\n\n".join(text_docs)}
"""
        }
        table_message: dict = {
            'type': 'text',
            'text': f"""
{"\n\n".join(table_docs)}
""" 
        }
        image_message: dict = {
            'type': 'text',
            'text': f"""
{"\n\n".join(imgs_docs)}
"""
        }
        messages: List[dict] = [table_message, text_message, image_message]
        return [HumanMessage(content=messages)]

    def answer_query(self, query: str) -> str:
        generate_answer_chain = (
            {'query': itemgetter('query'), 'retrieved_data': self.retriever_chain}
                |
            RunnableLambda(MultiModalRAG.create_query_context_prompt)
                | 
            gpt4o
                |
            StrOutputParser()
        )
        answer: str = generate_answer_chain.invoke({'query': query})
        return answer

query = "How much did LLaVA perform better?"
print("Query: ", query)
multi_modal_rag = MultiModalRAG("papers")
answer = multi_modal_rag.answer_query(query)
print("Answer: ", answer)

query = "What did LLaVA answer about the image of chair?"
print("Query: ", query)
answer = multi_modal_rag.answer_query(query)
print("Answer: ", answer)
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

from langchain_milvus import Milvus

# The easiest way is to use Milvus Lite where everything is stored in a local file.
# If you have a Milvus server you can use the server URI such as "http://localhost:19530".
URI = "./milvus_example.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
)

from langchain_core.documents import Document

vector_store_saved = Milvus.from_documents(
    [Document(page_content="foo!")],
    embeddings,
    collection_name="langchain_example",
    connection_args={"uri": URI},
)

vector_store_loaded = Milvus(
    embeddings,
    connection_args={"uri": URI},
    collection_name="langchain_example",
)





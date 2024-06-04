# Databricks notebook source
# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx]" langchain==0.2 llama-index==0.10.0 databricks-vectorsearch==0.22 pydantic==1.10.9 git+https://github.com/mlflow/mlflow.git
# MAGIC %pip install langchain-core langchain-openai tiktoken cloudpickle langchain-community azure-identity langchain-community
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

# Read the value of an RAG POC storage account and container/folder
poc_location_base = os.environ.get("LOCATION_BASE")

# Print the value
print(poc_location_base)

# COMMAND ----------

databricks_host = "https://"+spark.conf.get("spark.databricks.workspaceUrl")
personal_access_token = dbutils.secrets.get(scope="rag_poc_adb_scope", key="PAT")
#tenant_id = dbutils.secrets.get(scope="rag_poc_adb_scope", key="tenant_id")
#client_id = dbutils.secrets.get(scope="rag_poc_adb_scope", key="client_id")
#client_secret = dbutils.secrets.get(scope="rag_poc_adb_scope", key="client_secret")
# Azure login id is ApplicationID of "AzureDatabricks" Enterprise application in your Entra tenant
#login_id = dbutils.secrets.get(scope="rag_poc_adb_scope", key="azure_login_id")

for char in personal_access_token:
    print(char, end=" ")

# COMMAND ----------

dbutils.widgets.text("rag_poc_catalog", "rag_poc_catalog", "Catalog")
dbutils.widgets.text("rag_poc_schema", "rag_poc_schema", "Schema")
dbutils.widgets.text("rag_poc_volume_name", "ragpocvolume01", "rag_poc_volume_name")
dbutils.widgets.text("checkpoint_volume_name", "checkpoint01", "rag_poc_volume_name")
dbutils.widgets.text("rag_poc_deltaraw_name", "pdf_raw", "rag_poc_deltaraw_name")
dbutils.widgets.text("rag_poc_delta_name", "pdf_delta", "rag_poc_delta_name")
dbutils.widgets.text("rag_poc_vector_search_endpoint_name", "rag_poc_vector_search_endpoint", "rag_vector_search_endpoint_name")
dbutils.widgets.text("rag_poc_vector_search_index_name", "rag_poc_vector_search_index", "rag_vector_search_endpoint_name")
dbutils.widgets.text("model_name", "databricks-dbrx-instruct", "model_name")


# COMMAND ----------

import os

# Get the value of an environment variable
rag_poc_catalog = dbutils.widgets.get("rag_poc_catalog")
rag_poc_schema = dbutils.widgets.get("rag_poc_schema")
volume_name = dbutils.widgets.get("rag_poc_volume_name")
volume_location = poc_location_base+"/"+volume_name
checkpoint_volume_name = dbutils.widgets.get("checkpoint_volume_name")
checkpoint_location = poc_location_base+"/"+checkpoint_volume_name
raw_delta_name = dbutils.widgets.get("rag_poc_deltaraw_name")
raw_delta_location = poc_location_base+"/"+raw_delta_name
delta_name = dbutils.widgets.get("rag_poc_delta_name")
delta_location = poc_location_base+"/"+delta_name
vector_search_endpoint_name = dbutils.widgets.get("rag_poc_vector_search_endpoint_name")
vector_index_name = dbutils.widgets.get("rag_poc_vector_search_index_name")
model_name = dbutils.widgets.get("model_name")

# 
print (f"Volume location '{volume_location}' ")
print (f"Checkpoint location '{checkpoint_location}' ")
print (f"Raw delta table location '{raw_delta_location}' ")
print (f"Delta table location '{delta_location}' ")
print (f"model name '{model_name}' ")

# Create external volume that contains PDF
query = f"CREATE EXTERNAL VOLUME IF NOT EXISTS {rag_poc_catalog}.{rag_poc_schema}.{volume_name} LOCATION '{volume_location}'"
spark.sql(query)
# Create external volume for checkpoint
query = f"CREATE EXTERNAL VOLUME IF NOT EXISTS  {rag_poc_catalog}.{rag_poc_schema}.{checkpoint_volume_name} LOCATION '{checkpoint_location}'"
spark.sql(query)

# COMMAND ----------

query = f"""
CREATE TABLE IF NOT EXISTS {rag_poc_catalog}.{rag_poc_schema}.{raw_delta_name} (
   path string, 
   modificationTime TIMESTAMP, 
   length long, 
   content BINARY
) 
LOCATION '{raw_delta_location}'
"""
print(query)
# Run the query
spark.sql(query)

# COMMAND ----------

#query = f"""
#DELETE FROM {rag_poc_catalog}.{rag_poc_schema}.{raw_delta_name};
#"""
#spark.sql(query)

print(volume_name)

query = f"""
COPY INTO {rag_poc_catalog}.{rag_poc_schema}.{raw_delta_name} 
  FROM 'dbfs:/Volumes/{rag_poc_catalog}/{rag_poc_schema}/{volume_name}'
  FILEFORMAT = BINARYFILE
  PATTERN = '*.pdf';
"""
spark.sql(query)

query = f"""
SELECT * FROM {rag_poc_catalog}.{rag_poc_schema}.{raw_delta_name} LIMIT 1 ;
"""

df=spark.sql(query)
display(df)


# COMMAND ----------

from unstructured.partition.auto import partition
import re
import io

def extract_doc_text(x : bytes) -> str:
  # Read files and extract the values with unstructured
  sections = partition(file=io.BytesIO(x))
  def clean_section(txt):
    txt = re.sub(r'\n', '', txt)
    return re.sub(r' ?\.', '.', txt)
  return "\n".join([clean_section(s.text) for s in sections]) 




# COMMAND ----------

from llama_index.core.langchain_helpers.text_splitter import SentenceSplitter
from llama_index.core import Document, set_global_tokenizer
from transformers import AutoTokenizer
import io
import pandas as pd
from pyspark.sql.functions import col, pandas_udf
from typing import Iterator

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)
@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #set llama2 as tokenizer
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    def extract_and_split(b):
      txt = extract_doc_text(b)
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)




# COMMAND ----------

query = f"""
CREATE TABLE IF NOT EXISTS {rag_poc_catalog}.{rag_poc_schema}.{delta_name} (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY,
  url STRING,
  content STRING
) 
LOCATION '{delta_location}'
TBLPROPERTIES (delta.enableChangeDataFeed = true) 
"""
spark.sql(query)

# COMMAND ----------

from pyspark.sql.functions import explode
(spark.readStream.table(f"{rag_poc_catalog}.{rag_poc_schema}.{raw_delta_name}").repartition(4)
      .withColumn("content", explode(read_as_chunk("content")))
      .selectExpr('path as url', 'content')
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'dbfs:/Volumes/{rag_poc_catalog}/{rag_poc_schema}/{checkpoint_volume_name}')
    .table(f'{rag_poc_catalog}.{rag_poc_schema}.{delta_name}').awaitTermination())


# COMMAND ----------

query = f"""
SELECT * FROM {rag_poc_catalog}.{rag_poc_schema}.{delta_name} WHERE url like '%.pdf' limit 10;
"""
print(query)
df=spark.sql(query)
display(df)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import os
import json

#vsc = VectorSearchClient(workspace_url=databricks_host, service_principal_client_id=client_id, service_principal_client_secret=client_secret, azure_tenant_id=tenant_id, azure_login_id=login_id)
vsc = VectorSearchClient(workspace_url=databricks_host, personal_access_token=personal_access_token)

endpoints_response = vsc.list_endpoints()
endpoints = endpoints_response.get('endpoints', [])
endpoint_exists = any(endpoint['name'] == vector_search_endpoint_name for endpoint in endpoints)

if endpoint_exists:
    print(f"Endpoint '{vector_search_endpoint_name}' exists.")
else:
    vsc.create_endpoint(
        name=vector_search_endpoint_name,
        endpoint_type="STANDARD"
    )

# COMMAND ----------

# List all indexes for the specified endpoint
indexes_response = vsc.list_indexes(name=vector_search_endpoint_name)
indexes = indexes_response.get('vector_indexes', [])

# Check if your index exists
index_exists = any(index['name'] == f"{rag_poc_catalog}.{rag_poc_schema}.{vector_index_name}" for index in indexes)

if index_exists:
    print(f"Index '{rag_poc_catalog}.{rag_poc_schema}.{vector_index_name}' exists.")
else:
  index = vsc.create_delta_sync_index(
    endpoint_name=vector_search_endpoint_name,
    source_table_name=f"{rag_poc_catalog}.{rag_poc_schema}.{delta_name}",
    index_name=f"{rag_poc_catalog}.{rag_poc_schema}.{vector_index_name}",
    pipeline_type='TRIGGERED',
    primary_key="id",
    embedding_source_column="content",
    embedding_model_endpoint_name="databricks-bge-large-en"
  )

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

import os

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=databricks_host, personal_access_token=personal_access_token)
    #vsc = VectorSearchClient(workspace_url=databricks_host, service_principal_client_id=client_id, service_principal_client_secret=client_secret, azure_tenant_id=tenant_id, azure_login_id=login_id)
    vs_index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=f"{rag_poc_catalog}.{rag_poc_schema}.{vector_index_name}")

    # Create the retriever
    return DatabricksVectorSearch(vs_index).as_retriever()

# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("Responsibilities Director HR?")
print(f"Relevant documents: {similar_documents[0]}")



# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
from langchain.pydantic_v1 import BaseModel, Field
chat_model = ChatDatabricks(endpoint=model_name, max_tokens = 200)
print(f"Test chat model: {chat_model.invoke('What is Apache Spark')}")


# COMMAND ----------

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

retriever=get_retriever()

template = """
Assistant helps the company employees with their questions on company policies, roles. 
Always include the source metadata for each fact you use in the response. Use square brakets to reference the source, e.g. [role_library_pdf-10]. 
Properly format the output for human readability with new lines.
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever   , "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

#chain.invoke("where did harrison work?")
result = chain.invoke("Responsibilities of Director of Operations?")
print (result)

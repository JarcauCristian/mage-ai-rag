import chromadb

def retrieve_chroma_db_data():
    # Replace with your actual Chroma DB connection details
    chroma_client = chromadb.PersistentClient(path="./db")

    # Replace with your actual collection name
    collection_name = 'etl_pipelines_collection'

    # Retrieve the collection
    collection = chroma_client.get_collection(collection_name)

    # Fetch all records from the collection
    records = collection.get()

    # Output the records
    for record in records:
        print(records[record])

    # Close the connectio

if __name__ == '__main__':
    retrieve_chroma_db_data()

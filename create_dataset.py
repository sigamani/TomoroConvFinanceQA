import json
from tqdm.auto import tqdm
from langsmith import Client
from langchain_core.documents import Document
from config import DATA_PATH, DATA_LIMIT_EVAL, LANGFUSE_DATASET_NAME  # you may want to rename this var

def create_convfinqa_langsmith_dataset(filepath, name, description, limit: int = None) -> list[Document]:
    client = Client()

    # Create dataset
    dataset = client.create_dataset(
        dataset_name=name,
        description=description
    )

    with open(filepath, 'r') as f:
        data = json.load(f)

    QA_FIELDS = ["qa", *[f"qa_{i}" for i in range(10)]]
    SKIPPED_METADATA_FIELDS = ['annotation', *QA_FIELDS]

    if limit:
        data = data[:limit]

    examples = []

    for entry in tqdm(data):
        for qa_field in set(QA_FIELDS).intersection(entry.keys()):
            question = entry[qa_field]["question"]
            answer = entry[qa_field]["answer"]
            metadata = {
                "document": {field: entry[field] for field in entry.keys() if field not in SKIPPED_METADATA_FIELDS}
            }

            examples.append({
                "inputs": {"question": question},
                "outputs": {"answer": answer},
                "metadata": metadata,
            })

    # Upload all at once
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )

    print(f"✅ Created LangSmith dataset '{name}' with {len(examples)} examples.")

create_convfinqa_langsmith_dataset(
    DATA_PATH,
    LANGFUSE_DATASET_NAME,
    "Dataset created from ConvFinQA train data",
    limit=DATA_LIMIT_EVAL
)

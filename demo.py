from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import torch
import numpy as np

# Kết nối tới Milvus
connections.connect("milvus", host="localhost", port="19531")

# Định nghĩa schema của collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)  # Kích thước vector của mô hình BERT là 384
]
schema = CollectionSchema(fields, "Collection for text vectors")

# Tạo collection mới với tên "new_text_collection"
collection_name = "new_text_collection"
if utility.has_collection(collection_name):
    # Nếu collection đã tồn tại, xóa nó trước khi tạo mới
    collection = Collection(collection_name)
    collection.drop()

collection = Collection(name=collection_name, schema=schema)

# Khởi tạo mô hình và tokenizer
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Hàm để chuyển đổi văn bản sang vector
def encode(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Danh sách các văn bản cần chuyển đổi
texts = [
    "This is a sample sentence.",
    "Milvus is a vector database.",
    "Sentence Transformers are great for embedding texts.",
    "OpenAI's models are powerful.",
    "We are converting text to vectors."
]

# Chuyển đổi văn bản sang vector
vectors = [encode(text) for text in texts]

# Chèn dữ liệu vào collection (dữ liệu cần được đưa vào dưới dạng một danh sách duy nhất)
# Theo schema, danh sách đầu tiên là cho id (có auto_id=True, nên sẽ là None), danh sách thứ hai là cho vector
data = [None, np.array(vectors)]

collection.insert(data)

# Đảm bảo dữ liệu đã được lưu trữ
collection.flush()

# Tạo index để tối ưu hóa việc tìm kiếm
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}

collection.create_index(field_name="vector", index_params=index_params)

# Tải collection vào bộ nhớ để tìm kiếm nhanh hơn
collection.load()

# Thực hiện tìm kiếm với một câu truy vấn
query_text = "How to convert text to vectors?"
query_vector = encode(query_text).reshape(1, -1)

search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

results = collection.search(query_vector, "vector", param=search_params, limit=3, expr=None)

# In kết quả
for result in results:
    for match in result:
        print(f"Đối tượng ID: {match.id}, Khoảng cách: {match.distance}, Văn bản: {texts[match.id]}")

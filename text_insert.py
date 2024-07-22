from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, db, FieldSchema, CollectionSchema, DataType, Collection, utility
from docx import Document
import torch

# Kết nối tới Milvus
connections.connect("default", host="127.0.0.1", port="19531")

# Tạo database nếu chưa có
db.create_database("word")
connections.connect("default", host="127.0.0.1", port="19531", database="word")
db.list_database()
db.using_database("word")

# Tạo schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
]
schema = CollectionSchema(fields, "info")

# Tạo collection mới với tên "info"
collection_name = "info"
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

# Khởi tạo model và tokenizer từ Transformers
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Hàm để chuyển đổi văn bản sang vector
def encode(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Đọc nội dung từ file Word
def read_word_file(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Chia văn bản thành các câu dựa trên dấu chấm
def split_text_by_sentences(text):
    sentences = []
    current_sentence = ""
    for char in text:
        current_sentence += char
        if char == ".":
            sentences.append(current_sentence.strip())
            current_sentence = ""
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    return sentences

# Đường dẫn tới file trên máy
file_path = "D:\documents\your_document.docx"

# Đọc nội dung từ file Word
text = read_word_file(file_path)

# Chia văn bản thành các câu dựa trên dấu chấm
text_segments = split_text_by_sentences(text)

# Chuyển đổi từng câu văn bản sang vector và chèn vào collection
for segment in text_segments:
    vector = encode(segment)
    data = {
        "vector": [vector.tolist()],  # Chuyển vector thành list để Milvus có thể xử lý
        "text": [segment]
    }
    collection.insert([data["vector"], data["text"]])
    print(f"Inserted vector for text segment: {segment}")

# Đảm bảo dữ liệu đã được lưu trữ
collection.flush()

# Tạo index để tối ưu hóa việc tìm kiếm
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
if not collection.has_index():
    collection.create_index(field_name="vector", index_params=index_params)

# Tải collection vào bộ nhớ để tìm kiếm nhanh hơn
collection.load()


# Tìm kiếm
query_text = "Nam a?"
query_vector = encode(query_text).reshape(1, -1)

search_params = {"metric_type": "L2", "params": {"nprobe": 3}}

results = collection.search(query_vector, "vector", param=search_params, limit=3, expr=None, output_fields=["text"])

# In kết quả và lấy ID của dòng dữ liệu cần xóa
for result in results:
    for match in result:
        row_id_to_delete = match.id
        print(f"Đối tượng ID: {match.id}, Khoảng cách: {match.distance}, Text: {match.entity.get('text')}")

# Xóa dòng dữ liệu
expr = f"id in [{row_id_to_delete}]"
collection.delete(expr)
collection.flush()

# Xác nhận rằng dữ liệu đã bị xóa
results_after_delete = collection.search(query_vector, "vector", param=search_params, limit=1, expr=None, output_fields=["text"])

if len(results_after_delete) == 0 or results_after_delete[0].distances > 0.1:
    print("Dòng dữ liệu đã bị xóa thành công.")
else:
    print("Dòng dữ liệu vẫn tồn tại.")

from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, db, FieldSchema, CollectionSchema, DataType, Collection, utility
from docx import Document
import torch

# Kết nối tới Milvus
connections.connect("default", host="127.0.0.1", port="19531")

# Tạo database nếu chưa có
#db.create_database("word1")
connections.connect("default", host="127.0.0.1", port="19531", database="word1")
#db.list_database()
db.using_database("word1")

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
file_path = "D:/documents/LLM.docx"

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

#   collection
collection.load()

from transformers import AutoModelForCausalLM

llm_model_name = 'EleutherAI/gpt-neo-125M'
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)

# Hàm để đưa ra câu hỏi và lấy câu trả lời từ mô hình LLM
    def generate_qa_from_milvus(question):
        # Tìm kiếm dữ liệu từ Milvus
        query_vector = encode(question).reshape(1, -1)
        search_params = {"metric_type": "L2", "params": {"nprobe": 3}}
        results = collection.search(query_vector, "vector", param=search_params, limit=1, expr=None, output_fields=["text"])

        # Lấy context từ kết quả tìm kiếm và đặt câu hỏi để mô hình LLM trả lời
        for result in results:
            for match in result:
                context = match.entity.get("text")

                # Đặt câu hỏi và lấy câu trả lời từ mô hình LLM
                answer = generate_qa(question, context)
                print(f"Question: {question}")
                print(f"Answer: {answer}")

    # Hàm để đặt câu hỏi và lấy câu trả lời từ mô hình LLM
    def generate_qa(question, context):
        inputs = llm_tokenizer.encode(question + " [SEP] " + context, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = llm_model.generate(inputs, max_length=500, num_return_sequences=1, no_repeat_ngram_size=3)
        answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    # Đặt câu hỏi và lấy câu trả lời từ Milvus và mô hình LLM
    query_question = "what is the biometric?"
    generate_qa_from_milvus(query_question)



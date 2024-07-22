from transformers import AutoModel, AutoTokenizer
import torch
import os

# Định nghĩa đường dẫn lưu trữ
output_dir = "C:\\Users\\LATITUDE\\.cache\\huggingface\\hub\\sentence"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "all-MiniLM-L6-v2.onnx")

# Load model và tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tạo dummy input cho mô hình
dummy_input = tokenizer("This is a sample input", return_tensors="pt")

# Định nghĩa tên đầu vào và đầu ra cho mô hình ONNX
input_names = ["input_ids", "attention_mask"]
output_names = ["output"]

# Xuất mô hình sang định dạng ONNX
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    output_path,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
    opset_version=11
)

print(f"Model đã được chuyển đổi sang định dạng ONNX và lưu tại '{output_path}'")

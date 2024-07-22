import torch
from transformers import AutoModel, AutoTokenizer
import os

# Tải mô hình và tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Chuẩn bị input giả để chuyển đổi mô hình
dummy_input = tokenizer("This is a sample input", return_tensors="pt")

# Định nghĩa đường dẫn lưu trữ mô hình ONNX
onnx_model_path = r"C:\Users\LATITUDE\.cache\huggingface\hub\sentence\sentence-transformers-all-MiniLM-L6-v2.onnx"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

# Chuyển đổi mô hình sang ONNX với opset 14
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    onnx_model_path,
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=14  # Sử dụng phiên bản opset 14 hoặc cao hơn
)

print(f"Model has been converted to ONNX and saved at {onnx_model_path}")

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import spacy
from collections import defaultdict
import re
from sklearn.preprocessing import normalize

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

try:
    # nlp = spacy.load('/root/cqf/skeleton/en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

# 超参数, 可能需要调
beta = 0.5   
alpha = 0.9  
tau = 0.65    

# 数据预处理函数
def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())
    return tokens

def build_schema_vocab(tables, columns):
    schema_vocab = []
    schema_item_to_token_indices = defaultdict(list)  
    current_index = 0
    for item in tables + columns:
        tokens = preprocess(item)
        schema_vocab.extend(tokens)
        schema_item_to_token_indices[item].extend(range(current_index, current_index + len(tokens)))
        current_index += len(tokens)
    return schema_vocab, schema_item_to_token_indices

# 获取问句和模式的编码表示
def get_embeddings(text_tokens, schema_tokens):
    input_tokens = ['[CLS]'] + text_tokens + ['[SEP]'] + schema_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_tensor = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_tensor)
        last_hidden_state = outputs.last_hidden_state.squeeze(0)
    return last_hidden_state

# 余弦相似度计算函数
def cosine_similarity(u, v):
    u = u.numpy()
    v = v.numpy()

    u_normalized = u / np.linalg.norm(u, axis=1, keepdims=True)
    v_normalized = v / np.linalg.norm(v, axis=1, keepdims=True)
    similarity = np.dot(u_normalized, v_normalized.T)
    return similarity

# 名称匹配和值匹配
def token_matching(question_tokens, schema_items, schema_item_to_token_indices):
    Mm = np.zeros((len(question_tokens), len(schema_items)))
    for i, qt in enumerate(question_tokens):
        for j, item in enumerate(schema_items):
            if qt in preprocess(item):
                Mm[i][j] += 1.0 

    return Mm

# 词性标注
def pos_tagging(question_tokens):
    doc = nlp(' '.join(question_tokens))
    P = np.zeros(len(question_tokens))
    for i, token in enumerate(doc):
        if token.pos_ in ['NOUN', 'NUM', 'PROPN']:
            P[i] = alpha
    return P

def extract_question_skeleton(question, tables, columns):
    question_tokens = preprocess(question)
    schema_tokens, schema_item_to_token_indices = build_schema_vocab(tables, columns)
    schema_items = tables + columns
    n = len(question_tokens)
    m = len(schema_tokens)

    # 获取原始序列的表示
    embeddings = get_embeddings(question_tokens, schema_tokens)
    # 提取模式项的嵌入，不包括最后的 [SEP]
    h_s = embeddings[n + 2 : n + 2 + m]
    
    print("问句 token 数量 n =", n)
    print("模式 token 数量 m =", m)
    print("h_s shape:", h_s.shape)  # 应该是 (m, hidden_size)

    # 遍历每个问句 token，计算掩码后的表示并计算相似性
    Dp = np.zeros((n, m))
    for i in range(n):
        masked_question_tokens = question_tokens.copy()
        masked_question_tokens[i] = '[MASK]'
        masked_embeddings = get_embeddings(masked_question_tokens, schema_tokens)
        h_s_masked = masked_embeddings[n + 2 : n + 2 + m]

        # 计算余弦相似度
        similarity = cosine_similarity(h_s.cpu(), h_s_masked.cpu())
        Dp[i] = similarity.diagonal() 

    # 检查 Dp 的最大值以避免除以零
    max_Dp = Dp.max()
    if max_Dp == 0:
        max_Dp = 1.0  
    # 归一化 Dp
    Dp = Dp / max_Dp

    Mm = token_matching(question_tokens, schema_items, schema_item_to_token_indices)

    Dp_per_item = np.zeros((n, len(schema_items)))
    for j, item in enumerate(schema_items):
        token_indices = schema_item_to_token_indices[item]
        if len(token_indices) == 0:
            continue
  
        Dp_per_item[:, j] = Dp[:, token_indices].mean(axis=1)

    print("Dp_per_item shape:", Dp_per_item.shape) 
    print("Mm shape:", Mm.shape) 

    R = Dp_per_item + beta * Mm

    P = pos_tagging(question_tokens)

    Qsco = (R.sum(axis=1) / len(schema_items) + P) / 2

    print("Qsco:", Qsco)

    skeleton_tokens = [qt for i, qt in enumerate(question_tokens) if Qsco[i] > tau]

    skeleton_tokens = [token for token in skeleton_tokens if token != '[MASK]']
    if 'name' or 'number' in skeleton_tokens:
        skeleton_tokens = [token for token in skeleton_tokens if token != 'name' and token != 'number']
    print("问句骨架 tokens：", skeleton_tokens)

    pattern = r'\b(' + '|'.join(re.escape(kw) for kw in skeleton_tokens) + r')\b'
    question_skeleton = re.sub(pattern, "[Mask]", question, flags=re.IGNORECASE)

    return question_skeleton


if __name__ == '__main__':
    # test sample
    examples = [
        {
            "question": "What is the highest grade received in the Mathematics course?",
            "tables": ["Students", "Courses", "Grades"],
            "columns": ["StudentID", "CourseID", "CourseName", "Grade", "StudentName"]
        },
        {
            "question": "How many books are currently available in the library?",
            "tables": ["Books", "Authors", "Libraries"],
            "columns": ["BookID", "Title", "AuthorID", "LibraryID", "Availability"]
        },
        {
            "question": "Which movie has the highest rating on the platform?",
            "tables": ["Movies", "Ratings", "Users"],
            "columns": ["MovieID", "Title", "UserID", "Rating", "Genre"]
        },
        {
            "question": "Show the total revenue generated by each product category.",
            "tables": ["Products", "Orders", "Categories"],
            "columns": ["ProductID", "CategoryID", "OrderID", "Price", "Quantity"]
        },
        {
            "question": "List the top 5 customers with the most orders this month.",
            "tables": ["Customers", "Orders", "OrderDetails"],
            "columns": ["CustomerID", "OrderID", "OrderDate", "ProductID", "Quantity"]
        },
        {
            "question": "What is the most borrowed book from the library?",
            "tables": ["Books", "BorrowRecords", "Users"],
            "columns": ["BookID", "UserID", "BorrowDate", "ReturnDate", "Title"]
        },
        {
            "question": "How many patients visited the hospital last week?",
            "tables": ["Patients", "Appointments", "Doctors"],
            "columns": ["PatientID", "DoctorID", "AppointmentID", "Date", "Reason"]
        },
        {
            "question": "Which city has the highest number of restaurants?",
            "tables": ["Cities", "Restaurants"],
            "columns": ["CityID", "CityName", "RestaurantID", "RestaurantName"]
        },
        {
            "question": "Show the average time spent by users on the platform daily.",
            "tables": ["Users", "Sessions"],
            "columns": ["UserID", "SessionID", "StartTime", "EndTime", "Date"]
        },
        {
            "question": "What is the name of the project with the longest duration?",
            "tables": ["Projects", "Employees", "Assignments"],
            "columns": ["ProjectID", "EmployeeID", "AssignmentID", "StartDate", "EndDate"]
        }
    ]

    for example in examples:
        question = example["question"]
        tables = example["tables"]
        columns = example["columns"]
        skeleton = extract_question_skeleton(question, tables, columns)

        print("\n原始问句：", question)
        print("提取的骨架：", skeleton)
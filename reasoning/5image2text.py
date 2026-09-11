import pandas as pd
import requests
from PIL import Image
import io
import time
import ollama
from tqdm import tqdm
import torch
import base64

dataset = 'baby'
CSV_FILE_PATH = '../data/{}/meta-{}.csv'.format(dataset, dataset)
OUTPUT_CSV_PATH = '../data/{}/output_image2text.csv'.format(dataset)
IMURL_COLUMN = 'imUrl'
BATCH_SIZE = 10
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

def download_image(url, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            image_bytes = response.content
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            return base64_image
        except Exception as e:
            print(f"下载图片失败(尝试 {attempt + 1}/{max_retries}): {e}")
            time.sleep(2)  # 等待后重试
    return None

def generate_image_description(image, model_name="gemma3:27b"):
    try:
        response = ollama.generate(
            model=model_name,
            prompt="Please convert the given image into an accurate and concise textual description relevant to the Clothing, Shoes & Jewelry, focusing on extracting key attributes that can influence the buying behavior of users, such as color, material, style, functionality, etc. To generate the textual description using a one-paragraph natural language overview in no more than 100 words.",
            images=[image],
            # options={"num_predict": 50}
        )
        description = response['response'].split("\n")[-1].strip()
        return description
    except Exception as e:
        print(f"生成描述时出错: {e}")
        return None
    
def process_csv():
    df = pd.read_csv(CSV_FILE_PATH)
    if 'image_description' in df.columns:
        print("警告: 'image_description'列已存在，将被覆盖")
    df['image_description'] = None

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="处理进度"):
        batch = df.iloc[i:i+BATCH_SIZE]
        for idx, row in batch.iterrows():
            img_url = row[IMURL_COLUMN]
            
            if pd.isna(img_url):
                print(f"行 {idx}: 空图片URL，跳过")
                continue
                
            image = download_image(img_url)
            if image is None:
                print(f"行 {idx}: 无法下载图片 {img_url}")
                continue
                
            description = generate_image_description(image)
            if description:
                df.at[idx, 'image_description'] = description
                print(f"行 {idx}: image_description: {description}")
            else:
                print(f"行 {idx}: 生成描述失败")

        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"已保存进度到 {OUTPUT_CSV_PATH}")

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"处理完成! 结果已保存到 {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    process_csv()